"""
Credits: This work is derived from Aladdin Persson's work on Yolo(link below). We acknowledge and are grateful for his
contribution.
https://github.com/aladdinpersson/Machine-Learning-Collection.git
Youtube - https://www.youtube.com/watch?v=n9_XyCGr-MI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=5
"""
import os.path
import json
import torch
import torchvision.transforms as T
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from model import Yolov1_resnet
from dataset import VOCDataset
from loss import YoloLoss

# CSV_PATH = "./data/100examples.csv"
# CSV_PATH = "./data/8examples.csv"
CSV_PATH = "./data/train.csv"
IMG_DIR = "./data/images"
LABEL_DIR = "./data/labels"
CHKPOINT_PATH = "./chkpoint.pth"
LOG_FILE_PATH = "./console.log"

size_x = 224
size_y = 224
S = 19
B = 2
C = 20
BATCH_SZ = 32
NUM_WORKERS = 4
learning_rate = 0.0001
EPOCHS = 100
TRAIN_TEST_SPLIT = 0.95
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = T.Compose([T.Resize((size_x, size_y)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# transforms = T.Compose([T.Resize((size_x, size_y)),
#                         T.ToTensor()])


# def train_fn(train_loader, model, optimizer, loss_fn, epoch):
#     loop = tqdm(train_loader, leave=True)
#     mean_loss = []
#
#     for batch_idx, (x, y) in enumerate(train_loader):
#         x, y = x.to(DEVICE), y.to(DEVICE)
#         out = model(x)
#         loss = loss_fn(out, y)
#         mean_loss.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # update progress bar
#         loop.set_postfix(loss=loss.item())
#
#     # print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
#     # print("Epoch {0}/{1}\tloss {2:.5}".format(epoch, EPOCH, sum(mean_loss) / len(mean_loss)))

def load_model(CHKPOINT_PATH, model, optimizer):
    checkpoint = torch.load(CHKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss


def save_model(CHKPOINT_PATH, model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, CHKPOINT_PATH)


if __name__ == "__main__":
    logfile = open(LOG_FILE_PATH, "w")
    prev_epoch_mean_val_loss = float("inf")
    dataset = VOCDataset(CSV_PATH, IMG_DIR, LABEL_DIR, S, B, C, transforms)

    train_split = int(TRAIN_TEST_SPLIT * len(dataset))
    test_split = len(dataset) - train_split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_split, test_split],
                                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SZ,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    Yolo_model = Yolov1_resnet(S, B, C)
    Yolo_model.to(DEVICE)
    optimizer = optim.Adam(Yolo_model.parameters(), lr=learning_rate)
    loss_fn = YoloLoss(S, B, C)

    if os.path.exists(CHKPOINT_PATH):
        Yolo_model, optimizer, epoch, loss = load_model(CHKPOINT_PATH, Yolo_model, optimizer)
        start = epoch
    else:
        start = -1

    for epoch in range(start + 1, EPOCHS):
        start_time = time.time()
        mean_train_loss = []
        mean_val_loss = []

        # Train
        Yolo_model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out_train = Yolo_model(x)
            loss = loss_fn(out_train, y)
            mean_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_train_loss = sum(mean_train_loss) / len(mean_train_loss)

        # Cross Validation
        Yolo_model.eval()
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = Yolo_model(x)
            loss = loss_fn(out, y)
            mean_val_loss.append(loss.item())
        mean_val_loss = sum(mean_val_loss) / len(mean_val_loss)

        if mean_val_loss < prev_epoch_mean_val_loss:
            prev_epoch_mean_val_loss = mean_val_loss
            # save time by not saving on every epoch
            save_model(CHKPOINT_PATH, Yolo_model, optimizer, epoch, loss)

        # print("Epoch {0}/{1}\ttrain loss {2:.5}\tval loss {3:.5}\t{4} s"\
        #       .format(epoch, EPOCHS, mean_train_loss, mean_val_loss, int(time.time() - start_time)))

        logmsg="Epoch {0}/{1} train loss {2:.5} val loss {3:.5} {4} s"\
            .format(epoch, EPOCHS, mean_train_loss, mean_val_loss, int(time.time() - start_time))
        print(logmsg)
        json.dump(logmsg, logfile)
        logfile.write("\n")

    logfile.close()
