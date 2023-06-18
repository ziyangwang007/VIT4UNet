## ViT4UNet
Exploring CNN- and ViT-based Encoder-Decoder Network for Medical Image Segmentation.


## Requirements
* [Tensorflow 2.5.0+]
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, OpenCV......
* pip install -U segmentation-models==0.2.1
* pip install keras-unet-collection==0.0.7b0


## Dataset

1. CT Spine Segmentation (10 patients with around 500 slices each) 
The original CT Spine dataset is available to access at [Link](http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_2.3A_Spine_and_Vertebrae_Segmentation), and we provide preprocessed numpy format suitable to directly use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1Qe98-FUMpSkjF0gDi2PFeAlxq5TVi4LJ/view?usp=sharing)

2. CT COVID 19 Segmentation (around 1000 slices from >40 patients)
The original CT Spine dataset is available to access at Kaggle [Link](https://www.kaggle.com/competitions/covid-segmentation), and we provide preprocessed numpy format suitable to directly use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1kDhAdaxIz11oeYD6rtdSNtVTsqpESqXc/view?usp=sharing)

3. MRI Cardiac Segmentation (100 patients)
The original MRI Cardiac dataset is available to access at [Link](https://www.creatis.insa-lyon.fr/Challenge/acdc/), and we provide preprocessed h5 format to use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1F3JzBSIURtFJkfcExBcT6Hu7Ar5_f8uv/view?usp=drive_link)

4. Ultrasound Nerve Segmentation (6000 slices)
The original Ultrasound Nerve dataset is available to access at Kaggle [Link](https://www.kaggle.com/c/ultrasound-nerve-segmentation), and we provide preprocessed numpy format suitable to directly use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1-lmKpdhcA2ItKqnmQpygJITEEQf3EQqq/view?usp=sharing)


## Preprocessing Data

1. Noisy Label
```
python Process_Data_for_2D_NoisyLabel_Spine.py
```

2. Sparse(Scribble) Label
```
python Process_Data_for_2D_SparseLabel_numpy.py
```


## Usage
1. Clone the repo:
```
git clone https://github.com/ziyangwang007/VIT4UNet.git 
cd VIT4UNet
```
2. Download the pre-processed dataset

3. Train(15 encoder-decoder segmentation models) and test(dice, iou, accueacy, precision, sensitivity, specificity) the model.

```
python xxx.py
```

## Reference
```
@inproceedings{wang2021rar,
  title={RAR-U-Net: a residual encoder to attention decoder by residual connections framework for spine segmentation under noisy labels},
  author={Wang, Ziyang and Zhang, Zhengdong and Voiculescu, Irina},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  year={2021},
  organization={IEEE}
}

@inproceedings{wang2021quadruple,
  title={Quadruple augmented pyramid network for multi-class COVID-19 segmentation via CT},
  author={Wang, Ziyang and Voiculescu, Irina},
  booktitle={2021 43rd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgement

This code is mainly based on [keras_unet_collection](https://github.com/yingkaisha/keras-unet-collection), [segmentation_models](https://github.com/qubvel/segmentation_models), [CBAM](https://github.com/kobiso/CBAM-tensorflow).