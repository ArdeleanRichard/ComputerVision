import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.utils.data import Dataset

from object_detection.constants import device, learning_rate, checkpoint_file, ANCHORS, image_size, CELL_SIZE
from object_detection.model import YOLOv3
from object_detection.train import YOLOLoss
from object_detection.utils import load_checkpoint, convert_cells_to_bboxes, nms, plot_image
import albumentations as A

if __name__ == '__main__':
    test_transform = A.Compose(
        [
            # Rescale an image so that maximum side is equal to image_size
            A.LongestMaxSize(max_size=image_size),
            # Pad remaining areas with zeros
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
            ),
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

    # Taking a sample image and testing the model

    # Setting the load_model to True
    load_model = True

    # Defining the model, optimizer, loss function and scaler
    model = YOLOv3().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Loading the checkpoint
    if load_model:
        load_checkpoint(checkpoint_file, model, optimizer, learning_rate)

        # Defining the test dataset and data loader
    test_dataset = Dataset(
        csv_file="./data/pascal voc/test.csv",
        image_dir="./data/pascal voc/images/",
        label_dir="./data/pascal voc/labels/",
        anchors=ANCHORS,
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=True,
    )

    # Getting a sample image from the test data loader
    x, y = next(iter(test_loader))
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        # Getting the model predictions
        output = model(x)
        # Getting the bounding boxes from the predictions
        bboxes = [[] for _ in range(x.shape[0])]
        anchors = (
                torch.tensor(ANCHORS)
                * torch.tensor(CELL_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(device)

        # Getting bounding boxes for each scale
        for i in range(3):
            batch_size, A, S, _, _ = output[i].shape
            anchor = anchors[i]
            boxes_scale_i = convert_cells_to_bboxes(
                output[i], anchor, s=S, is_predictions=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    model.train()

    # Plotting the image with bounding boxes for each image in the batch
    for i in range(batch_size):
        # Applying non-max suppression to remove overlapping bounding boxes
        nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6)
        # Plotting the image with bounding boxes
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)