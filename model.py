"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3), # (kernel_size, #_filters, stride, padding)
    "M", # maxpool
    (3, 192, 1, 1),
    "M", # maxpool
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M", # maxpool
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M", # maxpool
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# Augment 1 conv layer, batch layer and leakyrelu layer as one CNN block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config #the yolo architecture (list) defined above
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels # self.in_channels = 3

        for x in architecture:
            if type(x) == tuple: # if it's NOT maxpooling
                layers += [
                    CNNBlock( 
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                        # in_channels = 3, x1 (out_channels) = 64 from (7, 64, 2, 3), kernel_size = 7 from (7, 64, 2, 3)...and so forth                
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list: # [(1, 256, 1, 0), (3, 512, 1, 1), 4]
                conv1 = x[0] # (1, 256, 1, 0)
                conv2 = x[1] # (3, 512, 1, 1)
                num_repeats = x[2] # 4

                for _ in range(num_repeats): # 0~3:
                    layers += [
                        CNNBlock(
                            in_channels, 
                            conv1[1], # 256 in (1, 256, 1, 0)
                            kernel_size=conv1[0], # 1 in (1, 256, 1, 0)
                            stride=conv1[2], # 1 in (1, 256, 1, 0)
                            padding=conv1[3], # 0 in (1, 256, 1, 0)
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1], # in channel, 256 in (1, 256, 1, 0)
                            conv2[1], # out channel, 512 in (3, 512, 1, 1)
                            kernel_size=conv2[0], # 3 in (3, 512, 1, 1)
                            stride=conv2[2], # 1 in (3, 512, 1, 1)
                            padding=conv2[3], # 1 in (3, 512, 1, 1)
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers) # unpack!

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        # split_size: the grid size, i.e. in yolo v1 it's 7x7
        # num_boxes: each grid will predict how many boxes. in yolo v1 it's 2
        # num_classes: how many class will be predicted, in pascal VOC it has 20 classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )


if __name__ == '__main__':
    
    S = 7
    B = 2
    C = 20
    
    model = Yolov1(split_size = S, num_boxes = B, num_classes = C)
    
    input_tensor = torch.randn((2, 3, 448, 448)) # [BATCH_SIZE, CHANNEL, HEIGHT, WIDTH]
    
    output = model(input_tensor)
    
    print(output.shape) # --> NOT RESHAPED YET!
    
    
    
    
    
    
    
    