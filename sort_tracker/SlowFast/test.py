import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from lib import slowfastnet
import torchvision
import cv2
import os
# put parameters of slowfast into config.py file
from config import params

def read_frames(filename):
    root = "/local/b/cam2/data/HumanBehavior/2"
    lst = open(filename, "r")
    file_list = []
    count = 0
    for f in lst:
        full = os.path.join(root, f)
        full = full.rstrip()
        # print(full)
        file_list.append(full)
        #if count >= 32:
        #    break
        count += 1
    
    buff = np.empty((1,len(file_list), 112, 112, 3), np.dtype('float32'))
    count = 0

    for full in file_list:
        img = cv2.imread(full)
        img = cv2.resize(img, (112,112))
        img = (img - 128.0)/128.0

        buff[0, count] = img
        count += 1
                
    buff = buff.transpose((0,4,1,2,3))
        
    return torch.from_numpy(buff)

def load_vid(filename):
    capture = cv2.VideoCapture(filename)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    buff = np.empty((1,frame_count, 112, 112, 3), np.dtype('float32'))
    count = 0
    ret = True
    while (count < frame_count and ret):
        ret, frame = capture.read()
        print(frame.shape)
        if not ret:
            continue 
        # get frame, resize
        new_frame = cv2.resize(frame, (112,112))  
        
        # normalize
        new_frame = (new_frame - 128.0)/128.0
        # place into a numpy array
        buff[0, count] = new_frame
        count += 1
   
    #print(buff.shape) 
    #print(buff[0])
    #print(buff[1])
    # normalize array
    # convert from [D, H, W, C] to [C, D, H, W] for pytorch
    # buff = buff.transpose((3,0,1,2))
    #norm = torchvision.transforms.Normalize(mean=0, std=0.1)
    #buff = norm(buff)
    buff = buff.transpose((0,4,1,2,3))
    print(buff.shape)

    return torch.from_numpy(buff)

def main():
    print("Loading SlowFast Model")

    model = slowfastnet.resnet50(class_num=params['num_classes'])

    # load pretrained weights into model
    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # set gpu stuff? TODO: figure out what this actually does
    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])

    # TODO: load a dataset
    tn = read_frames("names.txt")

    model.eval()
    output = model(tn)
    prediction = torch.argmax(output)
    print(output)
    print(prediction)
    


    # /local/b/cam2/data/HumanBehavior/slowfasttest/v_Biking_g09_c01.avi
    """
    testset = '/local/b/cam2/data/HumanBehavior/UCFsample'
    testset2 = '/local/b/cam2/data/HumanBehavior/UCFsample/validation/biking'
    video_files = []
    for r, d, f in os.walk(testset2):
        for file in f:
            if '.avi' in file:
                video_files.append(os.path.join(r, file))

    model.eval()
    for vid_file in video_files:
        data = load_vid(vid_file)
        output = model(data)
        prediction = torch.argmax(output)
        print(prediction)  

    """
                 
    # data = torch.randn(1,3,10,112,112)
    
    # data = load_vid('/local/b/cam2/data/HumanBehavior/slowfasttest/v_Biking_g04_c01.avi')
    """
    # data = DataLoader(VideoDataset(testset, mode='validation', clip_len=64,frame_sample_rate=1),batch_size=5, shuffle=False,num_workers=4)
    # print(data.shape)
    
     model.eval()
    for step, (inputs, label) in enumerate(data):
        # print(inputs.shape)
        output = model(inputs)
        print(torch.argmax(output))
    
        if step > 5:
            break
    """
    
    # get output from model
    #output = model(data)
    #print(output.shape)
    #print(output) 

    #prediction = torch.argmax(output)
    #print(prediction)


if __name__ == '__main__':
    main()
