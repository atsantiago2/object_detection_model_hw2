# # Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# # http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T



class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "drinks_imgs"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "drinks_masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "drinks_imgs", self.imgs[idx])
        mask_path = os.path.join(self.root, "drinks_masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # print('unique ids', obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # print('obj_ids', obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # print('masks:', masks)
        # print('masks:', masks.ndim)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

import downloader
def download_files():
    downloader.get_drinks_dataset()
    # downloader.get_model()


import os
import numpy as np
# from PIL import Image, ImageDraw
import dataload


def prepare_datasets(ignore_empty_mask = False):
    download_files()
    data_dir = 'data'
    # test_csv_path = os.path.join(data_dir, 'drinks', 'labels_test.csv')
    test_dict,  test_classes  = dataload.get_dataset_dict(train=False)

    # test_csv_path = os.path.join(data_dir, 'drinks', 'labels_train.csv')
    train_dict, train_classes = dataload.get_dataset_dict(train=True)

    if not ignore_empty_mask:
        # Remove No Content Samples
        keylist = []
        for key, val in train_dict.items():
            # print(len(val['masks']))
            if len(val['masks']) == 0:
                # print(key)
                keylist.append(key)
    
        print('Empty Masks Ignored\n', keylist)
        for key in keylist:
            del train_dict[key]
    
    destination_directory = os.path.join('data', 'drinks_imgs')
    images_exist = os.path.isdir(destination_directory)
    
    destination_directory = os.path.join('data', 'drinks_masks')
    masks_exist = os.path.isdir(destination_directory)

    if images_exist and masks_exist:
        print('Datasets Already Ready')
    else:
        print('Preparing Datasets')
        
        # Create Folders for Datasets
        dp = os.path.join('data', 'drinks_imgs')
        downloader.check_create_directory(dpath=dp)
        dp = os.path.join('data', 'drinks_masks')
        downloader.check_create_directory(dpath=dp)


        for key, val in train_dict.items():
            # print(key, val)
            masks = val['masks']

            # Load Image 
            img_path = val['filename']
            img = Image.open(img_path).convert("RGB")
            
            # Save File to Dir
            imfp = os.path.join('data', 'drinks_imgs')
            imfp = img_path.replace(os.path.join('data','drinks'), imfp)
            # print(imfp)
            img.save(imfp)

            #Generate Mask Image
            mask_list = []
            for mask in masks:
                # Convert Name to Num
                name = mask['Name']
                num = int(name)

                # Create Polygon Data
                xl = mask['x']
                yl = mask['y']
                poly = [(xc,yc) for xc,yc in zip(xl,yl)]
                poly

                #Draw Mask Layer Numpy
                img2 = Image.new('L', img.size, 0)
                draw = ImageDraw.Draw(img2).polygon(poly, outline=num, fill=num)
                mask_im = np.array(img2)
                mask_list.append(mask_im)
                # display(img3)

            # Combine Masks Into 1 Image 
            mask_list = np.maximum.reduce(mask_list)

            # print('\n\n', np.unique(mask_list))
            # print(mask_list.shape)

            # Create Image Mask into file
            im = Image.fromarray(mask_list)

            imfp = os.path.join('data', 'drinks_masks')
            imfp = img_path.replace(os.path.join('data','drinks'), imfp)
            imfp = imfp.replace('.jpg', '.png')
            # print(imfp)

            im.save(imfp)
            
            # print(im)

            
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0 # 0 For Windows, Can be increased in others

def main():
    prepare_datasets()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has 4 classes only - background , can, bottle, coke
    num_classes = 4
#     # use our dataset and defined transformations
    dataset = PennFudanDataset('data', get_transform(train=True))
    dataset_test = PennFudanDataset('data', get_transform(train=False))


    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

#     # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # PRETRAINED Model File Check
    modelfile_exist = False
    if downloader.filecheck('drinks_model_santiago.pth', 176220000):
        model.load_state_dict(torch.load('drinks_model_santiago.pth'))
        modelfile_exist = True

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    if modelfile_exist:
        '''If Model File Already Exists, Do not Train anymore'''
        print('\n\nModel File Exists\n SKIPPING Training, Direct to Evaluation')
        evaluate(model, data_loader_test, device=device)
        
    else:
        print('Model File Does NOT Exist\n Training Model')
        
        # let's train it for 10 epochs
        from torch.optim.lr_scheduler import StepLR
        num_epochs = 10

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

        # Save Model
        torch.save(model.state_dict(), "drinks_model_santiago.pth")


if __name__ == "__main__":
    main()
