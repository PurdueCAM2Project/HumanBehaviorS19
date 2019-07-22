# HumanBehaviorS19
A human behavior analyzer package

## About
This project is aimed at collecting data about human behavior through cameras. The current version is the initial implementation of the idea. The data collected includes location data for each person, based on the map provided, and actions performe by the person during those intervals. This data is present as a storyboard or a profile in the followuing format:

//Insert image containing the format of the profiles

## Working
!["MOT Pipeline"](https://github.com/PurdueCAM2Project/HumanBehaviorS19/blob/master/resources/pipeline.png)

*Fig. 1: The basic pipeline of the algorithm*

It starts by tracking multiple pedestrians using SORT (Simple Online Realtime Tracking). Then the frame is cropped to the dimensions of the bounding box containing each person, and is sent into an action detection algorithm (slow-fast network). ...

## Installation Instructions

Visit [installation.md](installation.md) for steps to install required dependencies, and to setup the repository.

## Usage

To use the sort tracker, use the following command

```
usage: python3 sort_tracker/main.py [-h] [-v] -i INPUT [-t OBJ_THRESH] [-n NMS_THRESH]  [-w] [--cuda] [--no-show]

Human Behavior Analysis

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input image or directory or video
  -t OBJ_THRESH, --obj-thresh OBJ_THRESH
                        objectness threshold, DEFAULT: 0.5
  -n NMS_THRESH, --nms-thresh NMS_THRESH
                        non max suppression threshold, DEFAULT: 0.4
                        output directory, DEFAULT: detection/
  -v, --video           flag for detecting a video input
  --cuda                flag for running on GPU
  -m, --map             flag for projecting detections on the map
  -c, --corr            flag for the input file containing the correspondance points
  ```

Example Usage, for tracking people in a video, and then projecting it onto a 2-D map: (When inside the sort_tracker directory)

```python3 main.py --cuda -v -i input_video.avi -m -j map.jpg -c corr_points.txt ```

The correspondance points (present in corr_points.txt) for mapping the tracks of each person onto a map (map.jpg here) is of the format:
<dl>
  <code>x<sub>11</sub> y<sub>11</sub> x<sub>12</sub> y<sub>12</sub></code>
  
  <code>x<sub>21</sub> y<sub>21</sub> x<sub>22</sub> y<sub>22</sub></code><br />
  <code>x<sub>31</sub> y<sub>31</sub> x<sub>32</sub> y<sub>32</sub></code><br />
  <code>...</code><br />
  <code>x<sub>n1</sub> y<sub>n1</sub> x<sub>n2</sub> y<sub>n2</sub></code>  
</dl>

Here, each row contains two x-y coordinate pairs. In the example above, there are n correspondance points. For row number n, for example,<dl><code>x<sub>n1</sub> y<sub>n1</sub></code></dl> are the x-y pair for a point in the video frame, and the <dl><code>x<sub>n2</sub> y<sub>n2</sub></code></dl> are the x-y pair for the same point in the image containing the map.


## Authors
[Mohamad Alani](https://github.com/moealani)

[Peter Huang](https://github.com/peterhuang88)

[Dhruv Swarup](https://github.com/dhruvswarup123)

[Chau Minh Nguyen](https://github.com/cnguyenm)

[Nourledin Hendy](https://github.com/nhendy)





