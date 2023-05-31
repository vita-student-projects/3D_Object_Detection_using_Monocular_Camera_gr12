
# SMGUP-net : slightly modified GUP-net 

As part of the course deep learning for autonomous vehicles, the class as a whole creates the
visualization interface of an autonomous vehicle. The project is broken down into sub-
sections addressing the different topics. This project focuses on 3D object detection using
a monocular camera. The goal is to recognize agent categories like cars, pedestrians and
cycles from a RGB monocular camera and identifying their size and location in space with
3D bounding boxes.



## Dataset and environement preparation

The code has been trained and tested using the [dataset KITT](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). To run the code the colored images, camera calibration metric and the training label are needed. As we had acces only to the training images of KITTI we had to unshuffeled the images as the images are taken from videos, separate them in training, validation and test and then reshuffel them. Like this we don't have any images from the same videos on the test and train dataset. You can find the images number of the test set here [test.txt]().

The environement can be setup using `pip install requirement.txt`

## file structur
```
 KITTI/
     ImageSets/
         test.txt
         train.txt
         val.txt
 data/
     calib/
     image_2/
     label_2/
     predicted/
```

The `label_2` folder contains the train, validation and test images.

The `predicted` folder is automatically created during inference.

The `ImageSets` folder contains the samples split indexes. Those files are given in the GitHub.

## Path configuration

To be able to run the code the path have to be change in two files: 

`code/experiment/config.yaml` the path to the KITTI folder have to be set (KITTI not included) and you can choose were you want to store the log and the output.

`code/main` the path to the code folder have to be set (code included).

## Trained model

We provied the weight of the trained model on [google Drive]() 

## Contribution

We implemented two main contribution. The first one is to add a curriculum learning on the first epochs based on training first on only the easy label, then only the medium one, then only the hard one and finally continue with all of them. Unfortunatly we didn't achieve better results for all the category, the cyclists and pedestrians are better but not the cars (see report for more information on implementation). The second is to add the detection of all the other labels of KITTI, the initial code detected only the cyclists, pedestrians and cars. The results of these different contribution can be seen below in AP3D.

|**model** | Cars<br>@IoU=0.7  | Cyclists<br>@IoU=0.5  | Pedestrians<br>@IoU=0.5 |
| :------------: | :------------: |:-------------:| :-----:|
||
|<font color="red">***easy***</font> |
|**GUP-net**           | 22.22 | 0.77 | 7.16 |
|**GUP-net curriculum** | 21.39 | 2.08 | 10.73 | 
|**GUP-net all labels**| 21.35 | 4.52 | 8.82 |  
|
|<font color="red">***medium***</font>|
|**GUP-net**           | 16.32 | 0.46 | 2.66 |
|**GUP-net curriculum** | 15.80 | 1.32 |   3.92 | 
|**GUP-net all labels**| 17.71 | 2.84 | 4.55 |  
|
| <font color="red">***hard***</font> |
|**GUP-net**           | 15.10| 0.46 | 2.22 |
|**GUP-net curriculum** | 14.46 | 1.38 |   3.33| 
|**GUP-net all labels**| 15.60 | 2.84 | 4.55 |   


## Evaluation and Visualization

To evaluate the model you have to run the compiled c file `evaluate_3D_AP40` with the folowing command line : 

`./evaluate_3D_AP40 "<your path to label_2(included)>" "<your path to predicted data(not included)"`

To visualize the images with the 2D and 3D predicted boxes you can use the matlab file `run_demo_test.m` and change the root_dir variable with your path to KITTI folder. Then the variable label_dir, image_dir and claib_dir should be correct but make sure it is the case.

## citation

    @article{lu2021geometry,
    title={Geometry Uncertainty Projection Network for Monocular 3D Object Detection},
    author={Lu, Yan and Ma, Xinzhu and Yang, Lei and Zhang, Tianzhu and Liu, Yating and Chu, Qi and Yan, Junjie and Ouyang, Wanli},
    journal={arXiv preprint arXiv:2107.13774},year={2021}}
