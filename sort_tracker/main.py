import torch
import cv2
import numpy as np
from torch.autograd import Variable
from yolo_resources.darknet import Darknet
from yolo_resources.util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
from datetime import datetime
from sort.sort import *

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def parse_args():
    parser = argparse.ArgumentParser(description='Human Behavior Analysis')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('-v', '--video', required=True, help='flag for adding a video input')
    parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
    parser.add_argument('-m', '--map', action='store_true',default=False, help='flag from projecting people on a map')
    parser.add_argument('-i', '--img', required=False, help='flag for adding a path to the image containing the map')
    parser.add_argument('-c', '--corr', required=False, help='correspondance points for the map projection as a .txt')

    args = parser.parse_args()

    return args

def draw_mot_bbox(img, bbox, colors, classes):
    #img = imgs[int(bbox[0])]
    # label = classes[int(bbox[-1])]
    labelInt = int(bbox[-1])
    label = "Object " + str(labelInt)
    p1 = tuple(bbox[0:2].int())
    p2 = tuple(bbox[2:4].int())

    color = colors[labelInt % len(colors)] if labelInt is not None else colors[0]
    
    cv2.rectangle(img, p1, p2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)
    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)


def detect_video(model, args):
    objDict = dict() # dict for all objects and frames
    
    if args.map == True:
        print("Mapping is on...")
        if (args.corr is None or args.img is None):
            print ("ERROR: BOTH the -c and -i flag required with the -m flag")
            exit()
        pts_src = np.empty([0,2])
        pts_dst = np.empty([0,2])

        with open(args.corr, "r") as f:
            txt = f.read()
            lines = txt.splitlines()
            for line in lines:
                splitLine = line.split(' ')
                pts_src = np.append(pts_src, np.array([[int(splitLine[0]), int(splitLine[2])]]) , axis=0)
                pts_dst = np.append(pts_dst, np.array([[int(splitLine[1]), int(splitLine[3])]]) , axis=0)
                
    
    #pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
    #pts_dst = np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])
    h_matrix, status = cv2.findHomography(pts_src, pts_dst)
    
           
               #imgcv = cv2.imread(args.map)
                
                #print('*********************')
                #print(pts_src)
                #print('-----------')
                #print(pts_dst)
                #print('*********************')



   # draw_bbox([frame], detection, colors, classes)
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]

    colors = pkl.load(open("yolo_resources/pallete", "rb"))
    classes = load_classes("yolo_resources/coco.names")
    #colors = [colors[1]]
    cap = cv2.VideoCapture(args.video)
    #output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #TODO: Change output path?
    out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))
    read_frames = 0

    mot_tracker = Sort()

    start_time = datetime.now()
    print('Detecting...')
    while cap.isOpened():
        retflag, frame = cap.read()
        read_frames += 1
        if retflag:
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()

            #print(detections.shape)


            #processresult changes the variable 'detections'
            # print(detections)
            #print(args.obj_thresh)
            #print(args.nms_thresh)
            detections = process_result(detections, 0.5, 0.4)
            cls_confs = detections[:, 6].cpu().data.numpy()
            cls_ids = detections[:, 7].cpu().data.numpy()

            if len(detections) != 0:
                detections = transform_result(detections, [frame], input_size)
                #print (detections)
                
                #for detection in detections:

                xywh = detections[:,1:5]
                '''
                xywh[:, 0] = (detections[:, 1] + detections[:, 3]) / 2
                xywh[:, 1] = (detections[:, 2] + detections[:, 4]) / 2

                # TODO: width and hieght are double what they're supposed to be and dunno why
                xywh[:, 2] = abs(detections[:, 3] - detections[:, 1]) *2
                xywh[:, 3] = abs(detections[:, 2] - detections[:, 4]) *2
                '''
                xywh = xywh.cpu().data.numpy() #-> THe final bounding box that can be replaced in the deepSort
                ######################################################
                #print(xywh.shape)
                #print(cls_confs.shape)
                #num_dets, temp = xywh.shape
                #Convert to MOT format
                xs = xywh[: , 0]
                ys = xywh[:, 1]
                ws = xywh[:, 2]
                hs = xywh[:, 3]
                #new_xs = xs - ws/2
                #new_ys = ys - hs/2

                MOT16_bbox = np.empty((0,5))


                for cls_id, cls_conf, x,y,w,h in zip(cls_ids, cls_confs, xs, ys, ws, hs):
                    #MOT16_temp = [read_frames, cls_id, x, y, w, h, cls_conf, -1, -1, -1]
                    MOT16_temp = [x, y, w, h, cls_conf]
                    np.set_printoptions(precision=2, linewidth=150)
                    MOT16_bbox = np.append(MOT16_bbox, [MOT16_temp], axis=0)
                """
                for i in range(num_dets):
                    # what exactly is read_frames
                    MOT16_temp = [xywh[i][0], xywh[i][1], xywh[i][2], xywh[i][3]]
                """
                
                #print("bboxinput: ", MOT16_bbox)
                #exit()
                tracking_boxes = mot_tracker.update(MOT16_bbox)

                #print("output: ", tracking_boxes)
                #print("-------------------NEW BOX-------------------------")
                for tracking_box in tracking_boxes:
                    
               
                    draw_mot_bbox(frame, torch.from_numpy(tracking_box), colors, classes)
                                        
                    if  args.map == True:
                        xMid = (tracking_box[0]+tracking_box[2]) / 2
                        y_bottom = (tracking_box[3]) 
                        a = np.array([[xMid, y_bottom]], dtype='float32')
                        a = np.array([a])
                        ret = cv2.perspectiveTransform(a, h_matrix)
                        print (ret)
                        #cv2.line(imgcv, ret, ret, colors[tracking_box[-1]%len(colors)])
                        exit()

                #print("------------------END BOX--------------------------")
            out.write(frame)

            if read_frames % 30 == 0:
                    print('Number of frames processed:', read_frames)
                    
                    # Save frames into folders
                    # for v, k in objDict.items():
                    #     i = 0
                    #     for ks in k:
                    #         path = "/home/shay/a/malani/cam2/HumanBehaviorS19/sort_tracker/" + str(v)
                    #         try: os.mkdir(path)
                    #         except: pass
                    #         img = cv2.imwrite(os.path.join(path , str(i)+'.jpg'), ks)
                    #         i+=1
        else:
            break

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    # print('MOT16_bbox: \n', MOT16_bbox)
    cap.release()
    out.release()

    print('Detected video saved to "output.avi"')

    return

def action_input():

    return

def main():

    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print('Loading network...')
    model = Darknet("yolo_resources/yolov3.cfg")
    #model.load_weights("yolo_resources/yolov3.weights")
    model.load_weights("/local/b/cam2/data/HumanBehavior/yolov3.weights")
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    detect_video(model, args)


if __name__ == '__main__':
    main()
