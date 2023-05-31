
# SMGUP-net : slightly modified GUP-net 

As part of the course deep learning for autonomous vehicles, the class as a whole creates the
visualization interface of an autonomous vehicle. The project is broken down into sub-
sections addressing the different topics. This project focuses on 3D object detection using
a monocular camera. The goal is to recognize agent categories like cars, pedestrians and
cycles from a RGB monocular camera and identifying their size and location in space with
3D bounding boxes.


# Run the code
## Dataset

The code has been trained and tested using the [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset. To run the code, the left color images, camera calibration matrices and the training labels are needed (~12Gb).



### file structure
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
```

The `ImageSets` folder contains the samples split indexes to take the samples in the data folder. Those files are given.

 The KITTI dataset is shuffled from the start. Having no access to the official testing set, we unshuffeled the images to combine them back in ordered sequences, separated them in training, validation and test sets and then shuffled back. Like this we don't have any images from the same videos on the test and train dataset. 

## Environement preparation
The required libraries can be installed with:

 ```yes | pip install -r requirements.txt```


## Path configuration

To be able to run the code, the path needs to be set in the configuration file: 

`SMGUP_net/experiment/config.yaml` the path to the KITTI folder has to be set (KITTI not included) and you can choose were you want to store the log and the output.

This configuration file controls all main parameters of the network.

## Run
By default, simply execute the code in training mode with:
```
python main.py
```
If the configuration is located at a specific location, execute:
```
python main.py --config <your_path_relative_to_main.py>
```
To run the code in evaluation/inference mode, set:
```
python main.py -e
```

## Pre-trained model

We provied the weight of the trained model on [google Drive](https://drive.google.com/drive/folders/15TaksCDFiUiWEr8XCPPyV8EDVX7YHvqQ?usp=sharing) 



# Contribution

We implemented two main contributions. The first one leverage the classification of KITTI's objects into three categories: easy, moderate and hard, depending on the occlusion level, the truncation and the pixel size. From that we apply a curriculum learning during the training epochs. First only the easy objects are trained, then only the moderates ones, then only the hard ones and finally all of them are used for the remaining epochs. Unfortunately we didn't achieve better results for all the category, the cyclists and pedestrians are better but not the cars (see the report for more information on implementation). 

The second contribution is to add the detection of all the other labels of KITTI, the initial code detecting only the cyclists, pedestrians and cars. The results of these different contribution can be seen below with the AP 3D metric.

|**model** | Cars<br>@IoU=0.7  | Cyclists<br>@IoU=0.5  | Pedestrians<br>@IoU=0.5 |
| :------------: | :------------: |:-------------:| :-----:|
||
|<font color="red">***easy***</font> |
|**GUP-net**           | 22.22 | 0.77 | 7.16 |
|**GUP-net curriculum** | 21.39 | 2.08 | 10.73 | 
|**GUP-net all labels**| 21.35 | 4.52 | 8.82 |  
||
|<font color="red">***medium***</font>|
|**GUP-net**           | 16.32 | 0.46 | 2.66 |
|**GUP-net curriculum** | 15.80 | 1.32 |   3.92 | 
|**GUP-net all labels**| 17.71 | 2.84 | 4.55 |  
||
| <font color="red">***hard***</font> |
|**GUP-net**           | 15.10| 0.46 | 2.22 |
|**GUP-net curriculum** | 14.46 | 1.38 |   3.33| 
|**GUP-net all labels**| 15.60 | 2.84 | 4.55 |   



## Evaluation and Visualization

To evaluate the model you have to run the compiled c file `evaluation_metrics/evaluate_3D_AP40` with the folowing command line : 

`./evaluate_3D_AP40 "<your path to label_2 folder (included)>" "<your path to predicted data folder (not included)"`

To visualize the images with the 2D and 3D predicted boxes you can use the matlab file `matlab_visualization/run_visualization.m` and change the `root_dir` variable with your path to the KITTI folder. Set also the path to the labels to show with the variable `label_dir`.

## Citation

    @article{lu2021geometry,
    title={Geometry Uncertainty Projection Network for Monocular 3D Object Detection},
    author={Lu, Yan and Ma, Xinzhu and Yang, Lei and Zhang, Tianzhu and Liu, Yating and Chu, Qi and Yan, Junjie and Ouyang, Wanli},
    journal={arXiv preprint arXiv:2107.13774},year={2021}}
