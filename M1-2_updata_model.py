import os
import numpy as np
import argparse
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.datasets
from tqdm import tqdm
from models.resnet_custom import resnet50_baseline
import torchvision.transforms as transforms
import openslide
from sklearn.metrics import roc_curve, auc
import pickle
import time
from Early_Stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description='Updating the feature extractor and patch classifier')
parser.add_argument('--datasetsName', type=str, default='tcga_er')
parser.add_argument('--model', type=str, default='AB-MIL', help='[AB-MIL, Trans-MIL, DS-MIL]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--nepochs', type=int, default=200, help='the maxium number of epochs to train')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_ids', type=int, nargs='+', default=[0])
parser.add_argument('--last_model', type=str, default=None)
parser.add_argument('--type', type=str, choices=['original', 'hardming','single'], default='original')
parser.add_argument('--round', type=int, default=1)  ##这里
parser.add_argument('--n_classes', type=int, default=2)
global args, best_acc
args = parser.parse_args()
args.results_dir = os.path.join("./data/results", args.datasetsName,args.model)
args.patch_save = os.path.join("./data/patchCuts", args.datasetsName,args.model)
args.device_ids = [args.device]

torch.cuda.set_device(args.device)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():
    best_acc = 0
    if args.type == 'single':
        model = Model(input_dim=1024, n_classes=args.n_classes)
    else:
        model = Model(input_dim=1024, n_classes=args.n_classes+1)
    # print(model)
    start = time.time()
    if args.last_model:
        ch = torch.load(args.last_model, map_location='cpu')
        model.load_state_dict(ch, strict=False)
        print("Successfully load weight.")


    model.to(device)
    model = nn.DataParallel(model, device_ids=args.device_ids)

    # normalization
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        normalize])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    model_save_dir = args.results_dir
    with open(os.path.join(model_save_dir, f't{args.round}_pseudo_label.pkl'), 'rb') as f:
        obj = pickle.load(f)
    train_dset_patch = obj['train_dset_patch']
    val_dset_patch = obj['val_dset_patch']

    # train_dset = Dataset(split=train_dset_patch, transform=trans)
    train_patch_path = os.path.join(args.patch_save,"round_"+str(args.round),'train')
    train_dset = torchvision.datasets.ImageFolder(root=train_patch_path,transform=trans)
    num_train = len(train_dset)
    # train_dset = ImageDataset(split=train_dset_patch, transform=trans)  ## if save in .jpg format
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_patch_path = os.path.join(args.patch_save,"round_"+str(args.round),'val')
    val_dset = torchvision.datasets.ImageFolder(root=val_patch_path, transform=trans)
    # val_dset = Dataset(split=val_dset_patch, transform=test_trans)
    # val_dset = ImageDataset(split=val_dset_patch, transform=test_trans)  ## if save in .jpg format
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # optimization
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    cudnn.benchmark = True

    model_save_path = os.path.join(model_save_dir, f't{args.round}_feature_extractor.pth')
    early_stopping = EarlyStopping(model_path=model_save_path,
                                   patience=10, verbose=True)


    for epoch in range(args.nepochs):
        print(f"epoch {epoch}:")
        train_loss, train_err = train(epoch, model, train_loader, criterion, optimizer,num_train)
        auc, val_err, val_loss = test(epoch, model, val_loader, criterion)
        print('Validating - Epoch: [{}/{}]\tLoss: {:.4f}\tACC: {:.4f}\tAUC: {:.4f}\t'.format(epoch + 1, args.nepochs,
                                                                                             val_loss,
                                                                                             1 - val_err, auc))

        early_stopping(epoch, val_loss, best_acc, model.module)
        if early_stopping.early_stop:
            print('Early Stopping')
            break
        print('\r')

    ch = torch.load(model_save_path, map_location='cpu')
    model.module.load_state_dict(ch, strict=False)
    auc, test_err, test_loss = test(0, model, val_loader, criterion)
    end = time.time()
    print('use time: ', end - start)
    print('Test\tLoss: {:.4f}\tACC: {:.4f}\tAUC: {:.4f}\t'.format(test_loss, 1 - test_err, auc))

def train(epoch, model, loader, criterion, optimizer,num_train):
    model.train()
    running_loss = 0.
    running_err = 0.
    for i, (img, label) in tqdm(enumerate(loader),total=int(num_train/args.batch_size)+1):
        optimizer.zero_grad()
        img = img.cuda()
        label = label.cuda()
        probs, _ = model(img)
        loss = criterion(probs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * label.size(0)
        ## binary classification
        err, fps, fns = errors(probs.detach(), label.cpu())
        running_err += err
    running_loss = running_loss / len(loader.dataset)
    running_err = running_err / len(loader.dataset)
    print('Training - Epoch: [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}\t'.format(epoch + 1, args.nepochs, running_loss,
                                                                          1 - running_err))
    return running_loss, running_err

def test(epoch, model, loader, criterion):
    model.eval()
    running_loss = 0.
    running_err = 0.
    probs = []
    labels = []

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.cuda()
            label = label.cuda()
            prob, _ = model(img)
            loss = criterion(prob, label)

            running_loss += loss.item() * label.size(0)
            err, fps, fns = errors(prob.detach(), label.cpu())
            running_err += err
            probs.extend(prob.detach()[:, 1].tolist())
            labels.extend(label.detach().tolist())
    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    running_loss = running_loss / len(loader.dataset)
    running_err = running_err / len(loader.dataset)
    return roc_auc, running_err, running_loss

def errors(output, target):
    _, pred = output.topk(1, 1, True, True)  # return the max index of output
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred != real
    err = float(neq.sum())
    fps = float(np.logical_and(pred == 1, neq).sum())
    fns = float(np.logical_and(pred == 0, neq).sum())
    return err, fps, fns

class Model(nn.Module):
    def __init__(self, input_dim=1024,n_classes=4):
        super(Model, self).__init__()
        self.backbone = resnet50_baseline(True)
        self.backbone.fc = nn.Linear(1024, n_classes)
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        _, feat = self.backbone(x)
        prob = self.fc(feat)
        return prob, feat

if __name__ == '__main__':
    main()