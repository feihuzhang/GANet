# GANet

[GA-Net: Guided Aggregation Net for End-to-end Stereo Matching](https://arxiv.org/pdf/1904.06587.pdf)

<img align="center" src="http://www.feihuzhang.com/GANet/GANet.jpg">

## Brief Introduction
We are formulating traditional geometric and optimization of stereo into deep neural networks ...

## Oral Presentation 

[Slides](http://www.feihuzhang.com/GANet/GANet.pptx), [Video](https://www.youtube.com/watch?v=tpyrxcGL_Zg&feature=youtu.be), Poster


## Building Requirements:

    gcc: >=5.3
    GPU mem: >=7G (for testing);  >=12G (for training, >=22G is prefered)
    pytorch: >=1.0
    tested platform/settings:
      1) ubuntu 16.04 + cuda 10.0 + python 3.6, 3.7
      2) centos + cuda 9.2 + python 3.7

## Notice:

Installing pytorch from source helps solve most of the errors (lib conflicts).

Please refer to https://github.com/pytorch/pytorch about how to reinstall pytorch from source.

## How to Use?

step 1: compile the libs by "sh compile.sh"

step 2: download and prepare the dataset

    download SceneFLow dataset: "FlyingThings3D", "Driving" and "Monkaa" (final pass and disparity files).
  
      -mv all training images (totallty 29 folders) into ${your dataset PATH}/frames_finalpass/TRAIN/
      -mv all corresponding disparity files (totallty 29 folders) into ${your dataset PATH}/disparity/TRAIN/
      -make sure the following 29 folders are included in the "${your dataset PATH}/disparity/TRAIN/" and "${your dataset PATH}/frames_finalpass/TRAIN/":
        
        15mm_focallength	35mm_focallength		A			 a_rain_of_stones_x2		B				C
        eating_camera2_x2	eating_naked_camera2_x2		eating_x2		 family_x2			flower_storm_augmented0_x2	flower_storm_augmented1_x2
        flower_storm_x2	funnyworld_augmented0_x2	funnyworld_augmented1_x2	funnyworld_camera2_augmented0_x2	funnyworld_camera2_augmented1_x2	funnyworld_camera2_x2
        funnyworld_x2	lonetree_augmented0_x2		lonetree_augmented1_x2		lonetree_difftex2_x2		  lonetree_difftex_x2		lonetree_winter_x2
        lonetree_x2		top_view_x2			treeflight_augmented0_x2	treeflight_augmented1_x2  	treeflight_x2	
	
    download and extract kitti and kitti2015 datasets.
        
Step 3: revise parameter settings and run "train.sh" and "predict.sh" for training, finetuning and prediction/testing.


## Pretrained models:

Pretrained models on sceneflow, kitti and kitti2015 datasets are avaiable at: (will update later)

| sceneflow (for fine-tuning, only 10 epoch) | kitti2012 (after fine-tuning) | kitti2015 (after fine-tuning)|
|---|---|---|
|[Google Drive](https://drive.google.com/open?id=1VkcBGkA_pXolgLhrWdpZPwfvzhQfWWJQ)|[Google Drive](https://drive.google.com/open?id=1WMfbEhzj-WLqYEI2jCH1YFUR6dYyzlVE)|[Google Drive](https://drive.google.com/open?id=19hVQXpcXwp7SrHgJ5Tlu7_iCYNi4Oj9u)|

## Results:

The results should be better than those reported in the paper.

## Reference:

If you find the code useful, please cite our paper:

    @inproceedings{zhang2019GANet,
      title={GA-Net: Guided Aggregation Net for End-to-end Stereo Matching},
      author={Zhang, Feihu and Prisacariu, Victor and Yang, Ruigang and Torr, Philip},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019}
    }
