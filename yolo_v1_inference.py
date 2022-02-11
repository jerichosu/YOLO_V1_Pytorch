#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:12:28 2022

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



# inference
DEVICE = 'cuda'
LOAD_MODEL_FILE = "overfit.pth.tar"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0


model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

model.to(DEVICE)
model.eval()

# %%

# some testing images
# path = 'two_dogs_test1.png'
# path = 'two_dogs_test2.jpg'
# path = 'three_dogs.png'
# path = 'yaz_test.jpg'
path = 'town_center.png'


img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # convert from numpy array to tensor
# inputs = torch.from_numpy(img) # torch.Size([H,W,3])
# inputs = inputs.permute(2,0,1) # torch.Size([3,H,W])
# print(inputs.shape)

# change the height and width to fit the model
transform=T.Compose([T.ToTensor(), # TO TENSOR FIRST!!!!!!!!!
                     T.Resize((448, 448))])
# when you use ToTensor() class, PyTorch automatically converts all images into [0,1].

inputs = transform(img) # torch.Size([3,448,448])
print(inputs.shape)

# increase the dimension
inputs = torch.unsqueeze(inputs, 0) # torch.Size([1,3,H,W])
print(inputs.shape)
        
input_tensor = inputs.to(DEVICE)

output = model(input_tensor)
print(output.shape)

# %%
#convert to results
bboxes = cellboxes_to_boxes(output) # convert to bboxes

# plot without non_max_suppression
# plot_image(input_tensor.squeeze(0).permute(1,2,0).to("cpu"), bboxes[0])
plot_image(img, bboxes[0])

# NMS
bboxes = non_max_suppression(bboxes[0], iou_threshold=0.8, threshold=0.15, box_format="midpoint")

# plot after NMS (this one plots the 448x448 results)
# plot_image(input_tensor.squeeze(0).permute(1,2,0).to("cpu"), bboxes)

# this one shows the actual image size
plot_image(img, bboxes)







# cap = cv2.VideoCapture(path)

# retaining = True
# kk = 0
# image_list = []

# clip = []
# while retaining:
#     retaining, frame = cap.read()
    
#     if not retaining and frame is None:
#         continue
#     # tmp_ = center_crop(cv2.resize(frame, (171, 128)))
#     # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
    
#     # i = 0
#     # i += 1
#     # if i == 0 and i % 7 == 0:
#     #     clip.append(frame)
        
#     clip.append(frame)
#     if len(clip) == 16:
#         inputs = np.array(clip).astype(np.float32)
        
#         # inputs = np.expand_dims(inputs, axis=0)
#         # inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
        
#         # convert from numpy array to tensor
#         inputs = torch.from_numpy(inputs) # torch.Size([16, 360, 640, 3])
#         inputs = inputs.permute(3,0,1,2) # torch.Size([3, 16, 360, 640])
#         # print(inputs.shape)
        
#         # change the height and width to fit the model
#         inputs = trans_val(inputs) # torch.Size([3, 16, 224, 224])
        
#         # increase the dimension
#         inputs = torch.unsqueeze(inputs, 0) # torch.Size([1, 3, 16, 224, 224])
        
        
#         # Data normalization
#         # Divide every pixel intensity by 255.0
#         inputs = inputs.type(torch.FloatTensor).div_(255.0)
#         # Pixel intensity from 0 ~ 255 to [-1, +1]
#         inputs = inputs.type(torch.FloatTensor).sub_(stats['mean'][:,None,None,None]).div_(stats['std'][:,None,None,None])
        
        
#         inputs = inputs.to(device)
#         # inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
#         with torch.no_grad():
#             outputs = model(inputs).squeeze(2)
#             # outputs = model.forward(inputs).squeeze(2)
            
#         # compute the probability    
#         m = nn.Softmax(dim=1)
#         # probs = torch.max(m(outputs))
#         probs_fight = m(outputs)[0][0]
        
#         # check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         # probs = torch.max(outputs)
#         # print(probs_fight)
#         predicted_labels = torch.argmax(outputs)
#         # _, predicted_labels = torch.max(outputs, 1)
#         # print(int(predicted_labels))
        
#         # print(outputs.shape)
        
#         # label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

#         cv2.putText(frame, labels[int(predicted_labels)], (5, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5,
#                     (0, 0, 255), thickness=2)
        
#         cv2.putText(frame, "Prob_fight: %.4f" % probs_fight, (5, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                     (0, 0, 255), thickness=2)
#         # print(len(clip))
#         clip.pop(0)
#         # print(len(clip))
#         # break
    
#     cv2.imshow('result', frame)
    
#     # make sure don't make gif too big
#     kk += 1
#     print(kk)
#     # if kk > 500 and kk < 800:
#     #     print(kk)
#     # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # image_list.append(frame_rgb)
        
#     # if kk > 800:
#     #     break
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     # cv2.waitKey(30)

# cap.release()
# cv2.destroyAllWindows()









