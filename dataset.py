"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image

from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None, # csv_file: either train.csv or test.csv on archive
    ):
        self.annotations = pd.read_csv(csv_file) #[000001.jpg, 000001.txt] stuff like that
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # labels
        # print(label_path) # ../../../archive/labels/2011_000763.txt ....
    
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])
        

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) #images
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        # print(image)
        # print(image.shape)
        # print(boxes.shape) # torch.size([N,5]), each row contains 1 bbox (and class) in the image

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)
            
        # put all boxes and it's corresponding class into the cube, 
        # so that each label.txt-->7x7x30 cube
        
        # Convert To Cells
        # create an empty tensor (7x7x30)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # 7 x 7 x (20+5*2)
        
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label) # convert class_label in box.tolist() from str to int 

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i # position relative to the (i,j) cell, 
            # for example: from computing i,j we know the cell (i,j) is responsible for predicting this bbox
            # x_cell, y_cell is the centroid of bbox in this (i,j) cell

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            # determins the height and width of the bbox on the (i,j) cell
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            
            
            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0: #label_matrix: 7x7x30 cube initialized previously
            # NOTE: HERE WE ONLY CARE (i,j) position, NOT ALL POSITION!!!!
            
                # Set that there exists an object
                label_matrix[i, j, 20] = 1 

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1 # set the class_label_th number to be 1
                # meaning something like 14th class is [0,0,0...1 (14th position), 0, 0, 0,|l_obj(20th), box_coord(21~25), zeros(25:end)]
            
            # print(image) #normalized already
            # print(image.shape)
        return image, label_matrix #note image here is NOT tensor yet!!!!!
    
    
    
if __name__ == '__main__':
    path = '../../../archive/train.csv'
    IMG_DIR = '../../../archive/images'
    LABEL_DIR = '../../../archive/labels'
    BATCH_SIZE = 1
    NUM_WORKERS = 8
    DEVICE = 'cuda'
    
    
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms
    
        def __call__(self, img, bboxes):
            for t in self.transforms:
                img, bboxes = t(img), bboxes
    
            return img, bboxes


    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    # when you use ToTensor() class, PyTorch automatically converts all images into [0,1].
    
    train_dataset = VOCDataset(
        path, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )
    
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              pin_memory=True, shuffle=True, drop_last=True,
                              )
    
    
    for x, y in train_loader:
        # x = x.to(DEVICE)
        # for idx in range(8):
        #     bboxes = cellboxes_to_boxes(model(x))
        #     bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #     plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
        print(x.shape)
        print(y.shape)
        # print(1)
        break
        
        
    
    
    
    
    
    
    
    
    
    
