import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image, ImageFile
from torchvision.models import detection

from object_detection.data import Dataset
from object_detection.model import YOLOv3
from object_detection.utils import iou, convert_cells_to_bboxes, nms, plot_image, save_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


from tqdm import tqdm
from object_detection.constants import class_labels, image_size, ANCHORS, device, batch_size, learning_rate, epochs, \
    save_model, CELL_SIZE


# Defining YOLO loss class
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        # Identifying which cells in target have objects
        # and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),
        )

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # Box prediction confidence
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]),
                               torch.exp(pred[..., 3:5]) * anchors
                               ], dim=-1)
        # Calculating intersection over union for prediction and target
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        # Calculating Object loss
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]),
                               ious * target[..., 0:1][obj])

        # Predicted box coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target box coordinates
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        # Calculating box coordinate loss
        box_loss = self.mse(pred[..., 1:5][obj],
                            target[..., 1:5][obj])

        # Claculating class loss
        class_loss = self.cross_entropy((pred[..., 5:][obj]),
                                        target[..., 5][obj].long())

        # Total loss
        return (
                box_loss
                + object_loss
                + no_object_loss
                + class_loss
        )


# Define the train function to train the model
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.cuda.amp.autocast():
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (
                    loss_fn(outputs[0], y0, scaled_anchors[0])
                    + loss_fn(outputs[1], y1, scaled_anchors[1])
                    + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)



if __name__ == '__main__':
    # Transform for training
    train_transform = A.Compose(
        [
            # Rescale an image so that maximum side is equal to image_size
            A.LongestMaxSize(max_size=image_size),
            # Pad remaining areas with zeros
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
            ),
            # Random color jittering
            A.ColorJitter(
                brightness=0.5, contrast=0.5,
                saturation=0.5, hue=0.5, p=0.5
            ),
            # Flip the image horizontally
            A.HorizontalFlip(p=0.5),
            # Normalize the image
            A.Normalize(
                mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
            ),
            # Convert the image to PyTorch tensor
            ToTensorV2()
        ],
        # Augmentation for bounding boxes
        bbox_params=A.BboxParams(
            format="yolo",
            min_visibility=0.4,
            label_fields=[]
        )
    )


    # Creating the model from YOLOv3 class
    model = YOLOv3().to(device)

    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Defining the loss function
    loss_fn = YOLOLoss()

    # Defining the scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Defining the train dataset
    train_dataset = Dataset(
        csv_file="./data/pascal voc/train.csv",
        image_dir="./data/pascal voc/images/",
        label_dir="./data/pascal voc/labels/",
        anchors=ANCHORS,
        transform=train_transform
    )

    # Defining the train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    # Scaling the anchors
    scaled_anchors = (
            torch.tensor(ANCHORS) *
            torch.tensor(CELL_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    # Training the model
    for e in range(1, epochs + 1):
        print("Epoch:", e)
        training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        # Saving the model
        if save_model:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")