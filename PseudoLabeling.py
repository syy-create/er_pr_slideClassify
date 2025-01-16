import h5py
import pickle
import argparse
import os
import torch
import numpy as np
import pandas as pd
import glob

parser = argparse.ArgumentParser(description='Pseudo labeling hardmining')
parser.add_argument('--datasetsName', type=str, default='thymus_plan1_ABB1B2')
parser.add_argument('--n_classes', type=int, default=3, help='num of classes for slide')
parser.add_argument('--model', type=str, default='AB-MIL', help='[AB-MIL, Trans-MIL, DS-MIL]')
parser.add_argument('--num_patches', type=int, default=50, help='num of classes for slide')
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()
args.dset = args.datasetsName
args.results_dir = os.path.join("./data/results", args.dset,args.model)
args.data_dir = os.path.join("./data/patches", args.dset)
args.label_csv = os.path.join("./data/label", args.dset,'label.csv')

def prepare_data(df):
    df_slide_id = df['slide_id'].tolist()
    df_label = df['label'].tolist()
    df_slide_id = [_slide_id.rstrip('.svs') for _slide_id in df_slide_id]
    slide_to_label = dict(zip(df_slide_id, df_label))
    return slide_to_label

model_save_dir = args.results_dir
with open(os.path.join(model_save_dir, f't{args.round}_primary_attn.pkl'), 'rb') as f:
    obj = pickle.load(f)
train_attns = obj['train_attns']
train_preds = obj['train_preds']
val_attns = obj['val_attns']
val_preds = obj['val_preds']
test_attns = obj['test_attns']
test_preds = obj['test_preds']

###预测结果
result_csv = pd.read_csv(os.path.join(model_save_dir,'detail',f"round_{str(args.round-1)}_result.csv"))
###数据标签
df = pd.read_csv(args.label_csv)
slide_to_label = prepare_data(df)

patch_num = args.num_patches * args.round

## training set
topmax_tumor = patch_num
topmax_normal = patch_num
topmin = patch_num
train_dset_patch = {}
for img_id, attn in train_attns.items():
    ###注意力得分
    attn = torch.from_numpy(attn)
    attn = attn[slide_to_label[img_id]]
    if args.round == 1:
        score = attn
    else:
        preds = torch.from_numpy(train_preds[img_id])
        preds = torch.transpose(preds, 1, 0)
        score = preds[slide_to_label[img_id]] * attn
    ###patch坐标对应
    h5py_path = os.path.join(args.data_dir, 'patches', img_id + '.h5')
    file = h5py.File(h5py_path, 'r')
    coord_dset = file['coords']
    coords = np.array(coord_dset[:])

    _, topmax_id = torch.topk(score, k=topmax_tumor, dim=0)
    _, topmin_id = torch.topk(-score, k=topmin, dim=0)

    label = [slide_to_label[img_id]] * topmax_id.size(0) + [args.n_classes] * topmin_id.size(0)
    idx = topmax_id.tolist() + topmin_id.tolist()

    topmax_id = topmax_id.numpy()
    topmin_id = topmin_id.numpy()

    topmax_coords = coords[topmax_id].tolist()
    topmin_coords = coords[topmin_id].tolist()

    select_coords = topmax_coords + topmin_coords


    # select_coords = topmax_coords
    train_dset_patch[img_id] = {'coords': select_coords, 'labels': label, 'idx': idx}


## validation set
topmax_tumor = patch_num
topmax_normal = patch_num
topmin = patch_num
val_dset_patch = {}
for img_id, attn in val_attns.items():
    ###注意力得分
    attn = torch.from_numpy(attn)
    attn = attn[slide_to_label[img_id]]
    if args.round == 1:
        score = attn
    else:
        preds = torch.from_numpy(val_preds[img_id])
        preds = torch.transpose(preds, 1, 0)
        score = preds[slide_to_label[img_id]] * attn

    ###patch坐标对应
    h5py_path = os.path.join(args.data_dir, 'patches', img_id + '.h5')
    file = h5py.File(h5py_path, 'r')
    coord_dset = file['coords']
    coords = np.array(coord_dset[:])

    _, topmax_id = torch.topk(score, k=topmax_tumor, dim=0)
    _, topmin_id = torch.topk(-score, k=topmin, dim=0)

    label = [slide_to_label[img_id]] * topmax_id.size(0) + [args.n_classes] * topmin_id.size(0)
    idx = topmax_id.tolist() + topmin_id.tolist()

    topmax_id = topmax_id.numpy()
    topmin_id = topmin_id.numpy()

    topmax_coords = coords[topmax_id].tolist()
    topmin_coords = coords[topmin_id].tolist()

    select_coords = topmax_coords + topmin_coords

    val_dset_patch[img_id] = {'coords': select_coords, 'labels': label, 'idx': idx}

new_obj = {
    'train_dset_patch': train_dset_patch,
    'val_dset_patch': val_dset_patch
}

with open(os.path.join(model_save_dir, f't{args.round}_pseudo_label.pkl'), 'wb') as f:
    pickle.dump(new_obj, f)
print('Finish')
