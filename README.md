# OCT Classification

## Dataset Context

Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time (Swanson and Fujimoto, 2017)

![fSTeZMd](C:\Users\sz\Desktop\医学图像处理\fSTeZMd.png)

(A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). (Middle right) Multiple drusen (arrowheads) present in early AMD. (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

## Dataset Content

The dataset is organized into 2 folders (train, test) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN). You can load OCT Dataset by click [**https://data.mendeley.com/datasets/rscbjbr9sj/2**](https://data.mendeley.com/datasets/rscbjbr9sj/2)

Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

## Structure

```python
../OCT_Classification
├── checkpoint
│   ├── oct2017_classification_finetune_resnet_10epochs.pth
│   └── PDBL_with_finetune_on_resnet.pkl
├── models
│   ├── pdbl.py
│   ├── resnet.py
│   ├── utils.py
│   └── __init__.py
├── OCT_Dataset
│   ├── OCT2017
│   │   ├── test
│   │   └── train
│   └── Processed_OCT2017
│       ├── test
│       └── train
├── utils
│   ├── dataset.py
│   ├── process.py
│   └── __init__.py
└── trainer.py
```

The role of each file or folder is as follows:

- `checkpoint` : save the parameters of models.
- `models` : architecture of the classifier models which is restnet50 with PDBL.
- `OCT_Dataset`: `OCT2017` is raw data and Processed_OCT2017 is processed data.
- `utils`: make Dataset  and preprocess the data.
- `trainer.py`: train and test model.

## Requirements

- matplotlib==3.3.4
- numpy==1.20.1
- opencv_contrib_python==4.5.4.60
- pandas==1.2.4
- scikit_learn==1.1.3
- torch==1.9.0
- torchvision==0.13.0
- tqdm==4.59.0

## Usage

### Installation

- Download the repository.

```python
git clone https://github.com/aishangcengloua/OCT_Classification.git
```

- Install python dependencies.

```python
pip install -r requirements.txt
```

## Training and Inference

```python
python trainer.py --save_dir checkpoint --train_dir OCT_Dataset/Processed_OCT2017/train --val_dir OCT_Dataset/Processed_OCT2017/test --n_classes 4 --train_model True
```

## Reference

- [**https://github.com/hhyx/OCT-classification**](https://github.com/hhyx/OCT-classification)
- [**https://ieeexplore.ieee.org/document/9740140**](https://ieeexplore.ieee.org/document/9740140)
- [**https://data.mendeley.com/datasets/rscbjbr9sj/2**](https://data.mendeley.com/datasets/rscbjbr9sj/2)