# Prediction of luminal-type breast cancer using deep learning via hematoxylin and eosin-stained whole-slide images
## Indouction
This code is used to predict the expression of er and pr in the whole slide of breast cancer.
## Dependecies
In the environment configuration, we use version Python==3.8, pytorch==1.10.0+cu102.
## Usage
Before starting training, we need to filter the background of the Whole slide at 20x magnification and cut it into non overlapping 256 * 256 patches in order to extract features using the pretrained ResNet50 model.
The h5 file recorded in [patches](https://github.com/syy-create/er_pr_slideClassify/tree/main/data/patches/er/patches) contains the coordinates of each patch at 40x magnification.
After the feature extraction is completed, we will put the extracted feature pt file into the [feature file](https://github.com/syy-create/er_pr_slideClassify/tree/main/data/feature/er)
Afterwards, we can start training by running the [sh](https://github.com/syy-create/er_pr_slideClassify/blob/main/demo/er%26pr/er.sh) on the terminal.

