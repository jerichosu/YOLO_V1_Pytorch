#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:09:29 2022

@author: 1517suj
"""


from model import Yolov1
import torch
import torch.optim as optim
import torchvision.transforms as T
import cv2
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes, #converting boxes from relative to the cell to relative to the entire image 
    get_bboxes,
    plot_image,
    load_checkpoint,)

# %%

# inference
DEVICE = 'cuda'
model_path = "yolov1_weights.pt"

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0

model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
model.load_state_dict(torch.load(model_path))

model.to(DEVICE)
model.eval()

# %%

# # some testing images
# # path = 'two_dogs_test1.png'
# # path = 'two_dogs_test2.jpg'
# # path = 'three_dogs.png'
# # path = 'yaz_test.jpg'
# path = 'town_center.png'
# # path = 'test_image.jpeg'


# img = cv2.imread(path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)

# # change the height and width to fit the model
# transform=T.Compose([T.ToTensor(), # TO TENSOR FIRST!!!!!!!!!
#                      T.Resize((448, 448))])
# # when you use ToTensor() class, PyTorch automatically converts all images into [0,1].

# inputs = transform(img) # torch.Size([3,448,448])
# print(inputs.shape)

# # increase the dimension
# inputs = torch.unsqueeze(inputs, 0) # torch.Size([1,3,H,W])
# print(inputs.shape)
        
# input_tensor = inputs.to(DEVICE)

# output = model(input_tensor)
# print(output.shape)

# # %%
# #convert to results
# bboxes = cellboxes_to_boxes(output) # convert to bboxes

# # plot without non_max_suppression
# plot_image(img, bboxes[0])

# # NMS
# bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.2, box_format="midpoint")

# # this one shows the actual image size
# plot_image(img, bboxes)


# %%

def plot_labelled_image(image, bboxes):
    
    color = (255, 0, 0)
    thickness = 2
    
    
    """Plots predicted bounding boxes on the image"""
    height, width, _ = image.shape

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in bboxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        
        lower_right_x = box[0] + box[2] / 2
        lower_right_y = box[1] + box[3] / 2
        
        ulx = int(upper_left_x*width)
        uly = int(upper_left_y*height)
        
        lrx = int(lower_right_x*width)
        lry = int(lower_right_y*height)
        
        start_point = (ulx, uly)
        end_point = (lrx, lry)
        
        cv2.rectangle(image, start_point, end_point, color, thickness)
        
    return image
        
        
        

# %%

# change the height and width to fit the model
transform=T.Compose([T.ToTensor(), # TO TENSOR FIRST!!!!!!!!!
                     T.Resize((448, 448))])


cap = cv2.VideoCapture(0)

retaining = True
kk = 0


color = (255, 0, 0)
thickness = 2


while retaining:
    retaining, frame = cap.read()
    
    height, width, _ = frame.shape
    # print(height)
    # print(width)
    
    if not retaining and frame is None:
        continue
    
    inputs = transform(frame) # torch.Size([3,448,448])
    
    # increase the dimension
    inputs = torch.unsqueeze(inputs, 0) # torch.Size([1,3,H,W])
    
    # to cuda
    input_tensor = inputs.to(DEVICE)
    
    # predictions 
    output = model(input_tensor)
    
    #convert to results
    bboxes = cellboxes_to_boxes(output) # convert to bboxes
    
    # NMS
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.2, box_format="midpoint")
    
    # print(bboxes)
    
    # labelled_image = plot_labelled_image(frame, bboxes)
    
    # Create a Rectangle potch
    for box in bboxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        
        lower_right_x = box[0] + box[2] / 2
        lower_right_y = box[1] + box[3] / 2
        
        ulx = int(upper_left_x*width)
        uly = int(upper_left_y*height)
        
        lrx = int(lower_right_x*width)
        lry = int(lower_right_y*height)
        
        start_point = (ulx, uly)
        end_point = (lrx, lry)
        
        cv2.rectangle(frame, start_point, end_point, color, thickness)
    
    
    
    cv2.imshow('result', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()










