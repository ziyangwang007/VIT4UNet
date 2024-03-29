## ViT4UNet
Exploring CNN- and ViT-based Encoder-Decoder Network for Medical Image Segmentation.


## Requirements
* Tensorflow 2.5.0+
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, OpenCV, etc


## Dataset

1. CT Spine Segmentation (10 patients with around 500 slices each) 

The original CT Spine dataset can be accessed [Link](http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_2.3A_Spine_and_Vertebrae_Segmentation). We provide a preprocessed numpy format suitable for direct use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1Qe98-FUMpSkjF0gDi2PFeAlxq5TVi4LJ/view?usp=sharing)

2. CT COVID 19 Segmentation (around 1000 slices from >40 patients)

The original CT Spine dataset can be accessed [Link](https://www.kaggle.com/competitions/covid-segmentation).  We provide a preprocessed numpy format suitable for direct use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1kDhAdaxIz11oeYD6rtdSNtVTsqpESqXc/view?usp=sharing)

3. MRI Cardiac Segmentation (100 patients)

The original MRI Cardiac dataset can be accessed [Link](https://www.creatis.insa-lyon.fr/Challenge/acdc/). We provide preprocessed h5 format to use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1F3JzBSIURtFJkfcExBcT6Hu7Ar5_f8uv/view?usp=drive_link)

4. Ultrasound Nerve Segmentation (6000 slices)

The original Ultrasound Nerve dataset can be accessed [Link](https://www.kaggle.com/c/ultrasound-nerve-segmentation).  We provide a preprocessed numpy format suitable for direct use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1-lmKpdhcA2ItKqnmQpygJITEEQf3EQqq/view?usp=sharing)

5. MRI Brain Tumor Segmentation 

The original MRI Brain Tumor dataset can be accessed [Link](https://www.med.upenn.edu/cbica/brats-2019/). We provide a preprocessed numpy format suitable for direct use in ViT4UNet at [Google Drive](https://drive.google.com/file/d/1erKoNzknobgn7gZYEXylsJFYqq-gc6xQ/view)




## Preprocessing Data

1. Noisy Label
```
python Process_Data_for_2D_NoisyLabel_Spine.py
```
![Example CT Spine, Ground Truth, Noisy Label](imgs/noisylabel.png)


2. Sparse(Scribble) Label
```
python Process_Data_for_2D_SparseLabel_numpy.py
```
![Example MRI Cardiac, Ground Truth, Scribble Label](imgs/sparselabel.png)



## Usage
1. Clone the repo:
```
git clone https://github.com/ziyangwang007/VIT4UNet.git 
cd VIT4UNet
```
2. Download the pre-processed dataset

3. Train(15 encoder-decoder segmentation models) the model.

```
python xxx.py
```

4. Test(dice, iou, accuracy, precision, sensitivity, specificity) the model

```
python evaluation.py
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

@inproceedings{wang2022triple,
  title={Triple-view feature learning for medical image segmentation},
  author={Wang, Ziyang and Voiculescu, Irina},
  booktitle={2022 MICCAI Workshop Resource-Efficient Medical Image Analysis},
  year={2022},
  organization={Springer}
}
```

## Acknowledgement

This code is mainly based on [keras_unet_collection](https://github.com/yingkaisha/keras-unet-collection), [segmentation_models](https://github.com/qubvel/segmentation_models), [CBAM](https://github.com/kobiso/CBAM-tensorflow), [3D-Dense-UNet](https://github.com/mrkolarik/3D-brain-segmentation).