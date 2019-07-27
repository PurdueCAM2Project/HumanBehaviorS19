# HumanBehaviorS19
A human behavior analyzer package

## About
This project is aimed at collecting data about human behavior through cameras. The current version is the initial implementation of the idea. The data collected includes location data for each person, based on the map provided, and actions performed by the person during those intervals. This output is returned as a storyboard or a profile in the following format:

//Insert image containing the format of the profiles

## Working
!["MOT Pipeline"](https://github.com/PurdueCAM2Project/HumanBehaviorS19/blob/master/resources/pipeline.png)

*Fig. 1: The basic pipeline of the algorithm*

It starts by tracking multiple pedestrians using SORT (Simple Online Realtime Tracking). Then the frame is cropped to the dimensions of the bounding box containing each person, and is sent into an action detection algorithm (slow-fast network). ...


## Dependencies and Packages

(The version numbers provided are the ones used while testing and development)

| Package Name  | Version No.   |
| ------------- | ------------- |
| OpenCV  | 4.1.0  |
|  NumPy | 1.15.4 |
|   SciPy| 1.1.0 |
|  FilterPy |1.1.0  |
|  Numba | 0.39.0 |
|   Scikit-Image | 0.14.0 |
|  Scikit-Learn | 0.19.2 |
|PyTorch|1.0.1|
|TorchVision|0.2.1|
|CudaToolKit|9.0|

## Installation Instructions

Visit [installation.md](installation.md) for steps to install required dependencies, and to setup the repository.

## Usage

To use the sort tracker, use the following command

```
usage: main.py [-h] [-t OBJ_THRESH] [-n NMS_THRESH] -v VIDEO [--cuda] [-m]
               [-i IMG] [-c CORR]

Human Behavior Analysis

optional arguments:
  -h, --help            show this help message and exit
  -t OBJ_THRESH, --obj-thresh OBJ_THRESH
                        objectness threshold, DEFAULT: 0.5
  -n NMS_THRESH, --nms-thresh NMS_THRESH
                        non max suppression threshold, DEFAULT: 0.4
  -v VIDEO, --video VIDEO
                        flag for adding a video input
  --cuda                flag for running on GPU
  -m, --map             flag from projecting people on a map
  -i IMG, --img IMG     flag for providing an imput map image to print the
                        tracking results on
  -c CORR, --corr CORR  correspondance points for the map projection as a .txt
  ```
  
NOTE: The -c flag is required with the -m flag, as it is necessary to generate the mapping. However the -i flag, along with the input image does not have to be provided. This -i flag is only for actually drawing the points onto the image of the map to get a visual representation, and is not necessary for generating the profiles


Example Usage, for tracking people in a video, and then projecting it and printing it onto a 2-D map: (When inside the sort_tracker directory)

```python3 main.py --cuda -v input_video.avi -m -i map.jpg -c corr_points.txt ```

The correspondance points (present in corr_points.txt) for mapping the tracks of each person onto a map (map.jpg here) is of the format:

<pre>
<code>x<sub>11</sub> y<sub>11</sub> x<sub>12</sub> y<sub>12</sub>
x<sub>21</sub> y<sub>21</sub> x<sub>22</sub> y<sub>22</sub>
x<sub>31</sub> y<sub>31</sub> x<sub>32</sub> y<sub>32</sub>
...
x<sub>n1</sub> y<sub>n1</sub> x<sub>n2</sub> y<sub>n2</sub></code>
</pre>

Here, each row contains two x-y coordinate pairs. In the example above, there are n correspondance points. For row number n, for example, <code>x<sub>n1</sub> y<sub>n1</sub></code> are the x-y pair for a point in the video frame, and the <code>x<sub>n2</sub> y<sub>n2</sub></code> are the x-y pair for the same point in the image containing the map.

## To Do:
- [ ] Removing cars from detections
- [ ] Retraining YOLO for small images of people
- [X] Integrating action detection
- [ ] Completing the description in the readme
- [X] Generation of Profiles
- [ ] Making a progress bar
- [ ] Obtaining timestamps online for storyboards 

## Authors
- [Mohamad Alani](https://github.com/moealani)
- [Peter Huang](https://github.com/peterhuang88)
- [Dhruv Swarup](https://github.com/dhruvswarup123)
- [Chau Minh Nguyen](https://github.com/cnguyenm)
- [Nourledin Hendy](https://github.com/nhendy)

## References
- Paper [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- Paper [Website](https://pjreddie.com/darknet/yolo/)
- YOLOv3 [Tutorial](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
- Paper [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- Paper [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
