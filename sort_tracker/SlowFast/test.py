import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from lib import slowfastnet
import cv2
# put parameters of slowfast into config.py file
from config import params

def load_vid(filename):
    capture = cv2.VideoCapture(filename)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    buff = np.empty((1,frame_count, 112, 112, 3), np.dtype('float32'))
    count = 0
    ret = True
    while (count < frame_count and ret):
        ret, frame = capture.read()
   
        # get frame, resize
        new_frame = cv2.resize(frame, (112,112))  

        # place into a numpy array
        buff[0, count] = new_frame
        count += 1
   
    #print(buff.shape) 
    #print(buff[0])
    #print(buff[1])
    # normalize array
    # convert from [D, H, W, C] to [C, D, H, W] for pytorch
    # buff = buff.transpose((3,0,1,2))
    buff = buff.transpose((0,4,1,2,3))
    print(buff.shape)
    #new_buff = np.zeros(1)
    #new_buff[0] = buff
     

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

    # /local/b/cam2/data/HumanBehavior/slowfasttest/v_Biking_g09_c01.avi
    

    # data = torch.randn(1,3,10,112,112)
    
    data = load_vid('/local/b/cam2/data/HumanBehavior/slowfasttest/v_Biking_g09_c01.avi')
    print(data.shape)

    
    # get output from model
    output = model(data)
    print(output.shape)
    print(output) 

    prediction = torch.argmax(output)
    print(prediction)


if __name__ == '__main__':
    main()
