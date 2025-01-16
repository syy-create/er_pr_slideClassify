import openslide
import os
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Updating the feature extractor and patch classifier')
parser.add_argument('--datasetsName', type=str,default='tcga_er')
parser.add_argument('--model', type=str, default='AB-MIL', help='[AB-MIL, Trans-MIL, DS-MIL]')
parser.add_argument('--slide_dir', type=str, default='/media/ipmi2023-sc/1.44.1-42962/suyunyi/tcga-brca', help='path to save wsi')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--type', type=str, choices=['original', 'hardming','single'], default='original')
parser.add_argument('--round', type=int, default=1)  ##这里
global args, best_acc
args = parser.parse_args()
args.results_dir = os.path.join("./data/results", args.datasetsName,args.model)
args.patch_save = os.path.join("./data/patchCuts", args.datasetsName,args.model)


###分配好的伪标签
with open(os.path.join(args.results_dir, f't{args.round}_pseudo_label.pkl'), 'rb') as f:
    obj = pickle.load(f)
train_dset_patch = obj['train_dset_patch']
val_dset_patch = obj['val_dset_patch']

###保存patch的路径
des_root_path = os.path.join(args.patch_save, "round_"+str(args.round))
os.makedirs(des_root_path,exist_ok=True)
for i in ["train",'val']:
    # for j in range(args.n_classes+1): #planB
    # for j in range(args.n_classes*2+1): #hardming
    if args.type == 'original':
        for j in range(args.n_classes + 1):  # original
            os.makedirs(os.path.join(des_root_path,i,str(j)),exist_ok=True)
    elif args.type == 'hardming':
        for j in range(args.n_classes * 2 + 1):
            os.makedirs(os.path.join(des_root_path, i, str(j)), exist_ok=True)
    elif args.type == 'single':
        for j in range(args.n_classes):
            os.makedirs(os.path.join(des_root_path, i, str(j)), exist_ok=True)
    else:
        raise ValueError

##desPath 指向train路径
desPath = os.path.join(des_root_path,'train')
print("extracting train patch")
for slideName,slidePseudo in tqdm(train_dset_patch.items()):
    slidePath = os.path.join(args.slide_dir, slideName+'.svs')
    if os.path.exists(slidePath):
        pass
    else:
        slidePath = os.path.join(args.slide_dir, slideName+'.ndpi')
    if os.path.exists(slidePath):
        pass
    else:
        slidePath = os.path.join(args.slide_dir, slideName+'.mrxs')
    coords = slidePseudo['coords']
    labels = slidePseudo['labels']
    ####读取patch
    slide = openslide.open_slide(slidePath)
    # print("procseeing_train", slidePath)
    for process_i,index in enumerate(coords):
        try:
            label = labels[process_i]
            img = slide.read_region(index,0,(512,512)).convert('RGB')
            img = img.resize((256,256))
            patch_save_path = os.path.join(desPath,str(label),slideName+"--"+str(index[0])+"_"+str(index[1])+".jpg")
            img.save(patch_save_path)
        except:
            print("有问题",slidePath)
            continue

###desPath 指向val路径
desPath = os.path.join(des_root_path,'val')
print("extracting val patch")
for slideName,slidePseudo in tqdm(val_dset_patch.items()):
    slidePath = os.path.join(args.slide_dir, slideName+'.svs')
    if os.path.exists(slidePath):
        pass
    else:
        slidePath = os.path.join(args.slide_dir, slideName+'.ndpi')
    if os.path.exists(slidePath):
        pass
    else:
        slidePath = os.path.join(args.slide_dir, slideName+'.mrxs')
    coords = slidePseudo['coords']
    labels = slidePseudo['labels']

    ####读取patch
    try:
        slide = openslide.open_slide(slidePath)
    except:
        print("切片有问题:",slidePath)
        continue
    # print("processing_val",slidePath)
    for process_i,index in enumerate(coords):
        try:
            label = labels[process_i]
            img = slide.read_region(index,0,(512,512)).convert('RGB')
            img = img.resize((256, 256))
            patch_save_path = os.path.join(desPath,str(label),slideName+"--"+str(index[0])+"_"+str(index[1])+".jpg")
            img.save(patch_save_path)
        except:
            print("有问题", slidePath)
            continue