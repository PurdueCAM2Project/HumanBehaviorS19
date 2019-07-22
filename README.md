# HumanBehaviorS19
A human behavior analyzer package

## About
This

## Working


## Installation Instructions

Visit [installation.md](installation.md) for steps to install required dependencies, and to setup the repository.

## Usage

To use the sort tracker, use the following command
`
usage: python3 sort_tracker/main.py [-h] -i INPUT [-t OBJ_THRESH] [-n NMS_THRESH] [-o OUTDIR]
                   [-v] [-w] [--cuda] [--no-show]

YOLOv3 object detection

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
`

Example Usage, for tracking people in a video, and then projecting it onto a 2-D map: (When inside the sort_tracker directory)
`
python3 main.py --cuda -v -i input_video.avi -m -c corr_points.txt 
`

## Authors
[Mohamad Alani](https://github.com/moealani)

[Peter Huang](https://github.com/peterhuang88)

[Dhruv Swarup](https://github.com/dhruvswarup123)

[Chau Minh Nguyen](https://github.com/cnguyenm)




