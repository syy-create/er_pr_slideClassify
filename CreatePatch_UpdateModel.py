import openslide
import os
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Updating the feature extractor and patch classifier')
parser.add_argument('--datasetsName', type=str, choices=['nsclc', 'pole'], default='nsclc')
parser.add_argument('--n_classes', type=int, default=2, help='num of classes for slide')
parser.add_argument('--model', type=str, default='AB-MIL', help='[AB-MIL, Trans-MIL, DS-MIL]')
parser.add_argument('--slide_dir', type=str, default='/home/ipmi2023/project/Data_NAS/open_datasets/TCGA_NSCLS_luad_lscc/WSI/mpp025', help='path to save wsi')
parser.add_argument('--patch_save', type=str, default='/media/ipmi2023/Data_18t/ipmi2023/data_ipmi2023_18t/BCL_log/TCGA_lung/Best_round1/patchCut', help='path to save patch')
parser.add_argument('--slideType', type=str, default='.svs')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--round', type=int, default=2)  ##这里
global args, best_acc
args = parser.parse_args()
args.results_dir = args.results_dir = os.path.join("./data/results", args.datasetsName,args.model)
