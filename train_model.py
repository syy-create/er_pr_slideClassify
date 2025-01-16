import csv
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset.dataloader import MyDataset
import torch.nn as nn
from models.cls_model_multi import BAL_P, BAL_A
from models.TransMIL import TransMIL
from models.model_dsmil import FCLayer, IClassifier, BClassifier, MILNet
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import pandas as pd
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, precision_score, \
    f1_score, recall_score
import pickle
import torch.nn.functional as F
from Early_Stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

##准备feature round_0的 label splits
parser = argparse.ArgumentParser(description='Updating the MIL classifier')
parser.add_argument('--datasetsName', type=str, default='', help='directory to save features')
parser.add_argument('--n_classes', type=int, default=2, help='num of classes for slide')
parser.add_argument('--model', type=str, default='AB-MIL', help='[AB-MIL, Trans-MIL, DS-MIL]')
parser.add_argument('--k', type=str, default=0, help='fold to ues')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--earlyStop', type=int, default=10)
###测试结果 超参
parser.add_argument('--is_test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=1000, help='the maxium number of epochs to train')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='learning rate')
parser.add_argument('--seed', type=int, default=10, help='random seed for reproducible experiment')
parser.add_argument('--encoding_size', type=int, default=1024)

args = parser.parse_args()
###feat_dir  数据的特征pt文件
args.feat_dir = os.path.join('./data/feature',args.datasetsName,"round_"+str(args.round))
###split_dir  训练验证测试划分csv文件
args.split_dir = os.path.join('./data/splits',args.datasetsName)
###results_dir
args.results_dir = os.path.join('./data/results',args.datasetsName,args.model)
os.makedirs(args.results_dir, exist_ok=True)
writer = SummaryWriter(f'./data/runs/{args.datasetsName}/lr{args.lr}_wd{args.weight_decay}')
torch.cuda.set_device(args.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def return_splits(csv_path):
    split_df = pd.read_csv(csv_path,dtype={'val':str,'test':str})
    train_id = split_df['train'].tolist()
    val_id = split_df['val'].dropna().tolist()
    test_id = split_df['test'].dropna().tolist()
    train_label = split_df['train_label'].tolist()
    val_label = split_df['val_label'].dropna().tolist()
    test_label = split_df['test_label'].dropna().tolist()
    train_split = dict(zip(train_id, train_label))
    val_split = dict(zip(val_id, val_label))
    test_split = dict(zip(test_id, test_label))
    return train_split, val_split, test_split

def train_epoch(epoch, model, optimizer, trainloader, criterion, measure=1, verbose=1):
    model.train()
    target_wsi_all = []
    attns = {}
    loss_nn_all = 0.
    for i_batch, sampled_batch in enumerate(trainloader):
        optimizer.zero_grad()  # zero the gradient buffer
        img_id, X, target = sampled_batch['img_id'], sampled_batch['feat'], sampled_batch['target']
        input = X.cuda()
        # target=target.cuda()
        target = target.type(torch.int64).cuda()
        logit, Y_prob, Y_hat, attn = model(input)
        target_wsi_all.append(target.item())
        attns[img_id[0]] = attn
        loss_cls_wsi = criterion(logit, target)
        loss = loss_cls_wsi
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        loss_nn_all += loss.detach().item()
    writer.add_scalar('Loss/train', loss_nn_all / len(trainloader.dataset), epoch)

    print("\nEpoch: {}, loss_all: {:.4f}".format(epoch, loss_nn_all / len(trainloader.dataset)))
    return attns

def BestTarget(predicted, target_wsi, threshold=-1):
    target_wsi = torch.tensor(target_wsi)
    predicted = torch.tensor(predicted)
    thresholds = torch.linspace(0,1,100)
    accs = []
    precisions = []
    recalls = []
    f1s = []
    conf_matrixs = []

    if threshold==-1:
        for threshold in thresholds:
            predicted_labels = (predicted[:,1] > threshold).int()
            acc = accuracy_score(target_wsi,predicted_labels)
            precision = precision_score(target_wsi,predicted_labels)
            recall = recall_score(target_wsi,predicted_labels)
            f1 = f1_score(target_wsi,predicted_labels)
            conf_matrix = confusion_matrix(target_wsi,predicted_labels)

            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            conf_matrixs.append(conf_matrix)
        bestACC_index = accs.index(max(accs))

    else:
        predicted_labels = (predicted[:, 1] > threshold).int()
        acc = accuracy_score(target_wsi, predicted_labels)
        precision = precision_score(target_wsi, predicted_labels)
        recall = recall_score(target_wsi, predicted_labels)
        f1 = f1_score(target_wsi, predicted_labels)
        conf_matrix = confusion_matrix(target_wsi, predicted_labels)

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        conf_matrixs.append(conf_matrix)
        bestACC_index=0

    print(f"best result conf matrix")
    print(f"{conf_matrixs[bestACC_index]}")
    print(f"best acc:{accs[bestACC_index]}  f1:{f1s[bestACC_index]}  precision:{precisions[bestACC_index]}   recal:{recalls[bestACC_index]}")
    return accs[bestACC_index], thresholds[bestACC_index]


def prediction(model, loader, criterion, testing="Val", threshold=-1):
    model.eval()

    total_loss = 0.
    total_wsi_loss = 0.
    logits = torch.Tensor().cuda()
    target_wsi = []
    attns = {}
    result_csv=[]

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(loader):
            img_id, X, target = sampled_batch['img_id'], sampled_batch['feat'], sampled_batch['target']
            input = X.cuda()
            target = target.type(torch.int64).cuda()
            logit, Y_prob, Y_hat, attn = model(input)
            result_csv.append([img_id[0],target.item(),Y_hat.item()])
            prob_list = Y_prob.tolist()[0]
            prob_list = [str(_) for _ in prob_list]
            result_csv[-1].extend(prob_list)
            target_wsi.append(target.item())
            logits = torch.cat((logits, logit), dim=0)
            attns[img_id[0]] = attn
            loss_cls_wsi = criterion(logit, target).item()
            loss = loss_cls_wsi
            total_loss += loss
            total_wsi_loss += loss_cls_wsi
    total_loss = total_loss / len(loader.dataset)
    total_wsi_loss = total_wsi_loss / len(loader.dataset)
    print(f"[{testing}]"," Loss: {:.4f} WSI Loss :{:.4f}".format(total_loss, total_wsi_loss))
    writer.add_scalar(f'Loss/{testing}', total_loss, epoch)
    output = F.softmax(logits, dim=1)
    score, pred = torch.max(output, dim=1)

    pred = pred.cpu().numpy()
    output = output.cpu().numpy()
    target_wsi = np.array(target_wsi)
    eq = np.equal(target_wsi, pred)
    print("ceshi",float(eq.sum())  , target_wsi.shape[0])
    acc = float(eq.sum()) / target_wsi.shape[0]

    if args.n_classes == 2:
        acc, threshold = BestTarget(output, target_wsi, threshold=threshold)
        print("threshold",threshold)
    else:
        print(confusion_matrix(target_wsi, pred))

    fpr, tpr, thresh = roc_curve(y_true=target_wsi, y_score=output[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    print(f"[{testing}]",'\t loss (all):{:.4f}'.format(total_loss), 'WSI acc: {:.4f}\t WSI auc: {:.4f}\t'.format(acc, roc_auc))
    torch.cuda.empty_cache()
    return total_loss, roc_auc, attns, result_csv, threshold

def patch_prediction(model, loader, criterion, testing=False):
    model.eval()
    logits = torch.Tensor()
    target_wsi = []

    instance_preds = {}
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(loader):
            img_id, X, target = sampled_batch['img_id'], sampled_batch['feat'], sampled_batch['target']
            input = X.cuda()
            target = target.type(torch.int64).cuda()
            logit, Y_prob, Y_hat = model(input)
            target_wsi.append(target.item())
            # logits = torch.cat((logits, Y_prob.detach().cpu()), dim=0)
            instance_preds[img_id[0]] = Y_prob.detach().cpu().numpy()
    # score, pred = torch.max(logits, dim=1)
    torch.cuda.empty_cache()
    return instance_preds

if __name__ == '__main__':
    batch_size = args.batch_size
    num_epochs = args.epochs
    feat_dir = args.feat_dir
    lr = args.lr
    weight_decay = args.weight_decay
    test_accs = []
    model_save_dir = args.results_dir

    ###数据准备
    train_dataset, val_dataset, test_dataset = return_splits(csv_path='{}/fold{}.csv'.format(args.split_dir, args.k))
    train_dset = MyDataset(feat_dir=args.feat_dir, train=True, split=train_dataset)
    trainloader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
    val_dset = MyDataset(feat_dir=args.feat_dir, train=False, split=val_dataset)
    valloader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)
    test_dset = MyDataset(feat_dir=args.feat_dir, train=False, split=test_dataset)
    testloader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)

    ###模型选择
    if args.model == "AB-MIL":
        model = BAL_P(n_classes=args.n_classes, input_dim=args.encoding_size, subtyping=False).to(device)
    elif args.model == "Trans-MIL":
        model = TransMIL(n_classes=args.n_classes, device=device).to(device)
    elif args.model == "DS-MIL":
        i_classifier = FCLayer(in_size=args.encoding_size, out_size=args.n_classes)
        b_classifier = BClassifier(input_size=args.encoding_size, output_class=args.n_classes, dropout_v=0.0)
        model = MILNet(i_classifier, b_classifier).to(device)
    else:
        raise ValueError(f"{args.model} model not implement!")

    ###参数设置

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    best_acc = 0

    model_path = os.path.join(model_save_dir, f't{args.round}_primary.pth')
    early_stopping = EarlyStopping(model_path=model_path, patience=args.earlyStop, verbose=True)

    ###预加载权重
    # model.load_state_dict(torch.load(""))

    if not args.is_test:
        ## training the MIL classifier
        for epoch in range(args.epochs):
            train_attns = train_epoch(epoch, model, optimizer, trainloader, criterion)
            valid_loss, val_auc, val_attns, _, _ = prediction(model, valloader, criterion)
            early_stopping(epoch, valid_loss, best_acc, model)
            if early_stopping.early_stop:
                print()
                print('Early Stopping')
                break
            print('\r')
    writer.close()
    trainloader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=8)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    ###测试并保存结果
    result_csv = [["slide_id", 'label', 'pre']]
    for _ in range(args.n_classes):
        result_csv[0].append("prob_"+str(_))
    _, _, val_attns, val_result,threshold  = prediction(model, valloader, criterion, testing="Val")
    _, test_acc, test_attns, test_result,_ = prediction(model, testloader, criterion, testing="Test", threshold=0.5)
    _, _, train_attns, train_result, _ = prediction(model, trainloader, criterion, testing="Train")
    result_csv.append(['train'])
    result_csv.extend(train_result)
    result_csv.append(['val'])
    result_csv.extend(val_result)
    result_csv.append(['test'])
    result_csv.extend(test_result)
    os.makedirs(f"{args.results_dir}/detail/", exist_ok=True)
    with open(f"{args.results_dir}/detail/round_{str(args.round)}_result.csv", 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(result_csv)

    ###保存attention
    if not args.is_test:
        if args.round == 0:
            test_preds, train_preds, val_preds = None, None, None
        else:
            model = BAL_A(n_classes=args.n_classes, input_dim=args.encoding_size).cuda()
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
            test_preds = patch_prediction(model, testloader, criterion, testing=True)
            train_preds = patch_prediction(model, trainloader, criterion, testing=True)
            val_preds = patch_prediction(model, valloader, criterion, testing=True)

        obj = {
            'train_attns': train_attns,
            'train_preds': train_preds,
            'val_attns': val_attns,
            'val_preds': val_preds,
            'test_attns': test_attns,
            'test_preds': test_preds
        }
        with open(os.path.join(model_save_dir, f't{args.round + 1}_primary_attn.pkl'), 'wb') as f:
            pickle.dump(obj, f)
