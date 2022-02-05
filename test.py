from display import disp
import os.path
import sys
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import Yolov1_resnet
from dataset import VOCDataset
from PIL import Image

CSV_PATH = "./data/8examples.csv"
IMG_DIR = "./data/images"
LABEL_DIR = "./data/labels"
CHKPOINT_PATH = "./chkpoint.pth"
size_x = 224
size_y = 224
S = 7
B = 2
C = 20
BATCH_SZ = 1
NUM_WORKERS = 1
learning_rate = 0.0001
EPOCHS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = T.Compose([T.Resize((size_x, size_y)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

invTrans = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                        T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path>")
        sys.exit()
    else:
        img_path = str(sys.argv[1])
        if not os.path.exists(img_path):
            print("Invalid path - {0}".format(img_path))
            sys.exit()

    display = disp()
    model = Yolov1_resnet(S, B, C)
    if not os.path.exists(CHKPOINT_PATH):
        print("Checkpoint file not found. Exiting!")
        sys.exit()
    checkpoint = torch.load(CHKPOINT_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    from matplotlib import pyplot as plt
    image = Image.open(img_path).resize((size_x, size_y))
    X = transforms(image)
    X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2]).to(DEVICE)
    y = model(X)
    y = y.detach().reshape((-1, S, S, (C + 5 * B)))
    display.show(image, y)