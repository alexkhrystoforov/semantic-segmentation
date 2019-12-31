# Semantic segmentation with Keras/Tensorflow using U-net neural network

### Files:

EDA.ipynb - Jupyter notebook with exploratory data analysis

unet.py - our U-net model

dsb.py - some classes and scripts

train_and_predicted_masks.py - the script that will run training and show results

datasets - folder which contains dataset

## Dataset - 2018 Data Science Bowl

Data set from kaggle competition:
https://www.kaggle.com/c/data-science-bowl-2018/data

We took only train set. Divie images by 8 clusters (according from their statistic) and then
create CV and test set with equal proportions of different clusters.

Dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.

Each image is represented by an associated ImageId. Files belonging to an image are contained in a folder with this ImageId. Within this folder are two subfolders:

- images contains the image file.
- masks contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one nucleus. Masks are not allowed to overlap (no pixel belongs to two masks).

### Image sizes

There is a large range of image dimensions in the dataset and not all of the images are square. The smallest image was 256x256 (which we will take as a basis size) and the largest was 1040x1388 pixels.

The smallest nucleus was only a few pixels in size and was found in one of the larger images (1000x1000).

The main idea is - decompose images that are larger than our basis by overlapped 256x256 tiles

Overlapping of tiles brings another positive effect: it increases the size of the training set.

### Images color

A lot of images from training set have different color - so we convert them to grayscale

## Model

The architecture used is [U-Net](https://arxiv.org/abs/1505.04597), which is very common for image segmentation problems.

Our metric is [Dice Sore](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2).

Training of the model was done using Adam optimizer with learning rate 1.e-4.

The model is trained for 3 (3500 steps on each) epochs.

After 5 epochs we get Dice Score - 0.602.
#### Links:

https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering

https://medium.com/analytics-vidhya/semantic-segmentation-using-u-net-data-science-bowl-2018-data-set-ed046c2004a5

https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5