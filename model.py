"""
Credits: This work is derived from Aladdin Persson's work on Yolo(link below). We acknowledge and are grateful for his
contribution.
https://github.com/aladdinpersson/Machine-Learning-Collection.git
Youtube - https://www.youtube.com/watch?v=n9_XyCGr-MI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=5

Object detection based on Yolo and Yolo datasets  from 2006-2012.
"""
import torch
import torch.nn as nn
from torchvision import models

class Yolov1_resnet(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(Yolov1_resnet, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        # self.nn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.nn = models.resnet18(pretrained=True)
        # self.nn1 = nn.Sequential(*list(self.nn.modules())[:-1])
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        self.nn.fc = nn.Linear(in_features=512, out_features=1024 * S * S, bias=True)
        self.nn.fc1 = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(1024 * S * S, 496),
                        nn.Dropout(0.0),
                        nn.LeakyReLU(0.1),
                        nn.Linear(496, S * S * (C + B * 5)),
                    )

    def forward(self, batch):
        out = self.nn(batch)
        out = self.nn.fc1(out)
        return out



if __name__ == "__main__":
    my_model = Yolov1_resnet(9, 2, 3)
    print(my_model.nn)
    print("___________________DONE__________________")
    images = torch.zeros((32, 3, 224, 224))
    print(images.shape)

    out = my_model(images)
    print(out.shape)



