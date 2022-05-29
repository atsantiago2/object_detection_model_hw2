
from ast import Raise
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


import downloader
def download_files():
    downloader.get_model()

import os
import numpy as np
# from PIL import Image, ImageDraw


TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0 # 0 For Windows, Can be increased in others

def main():
	download_files()
	
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# our dataset has 4 classes only - background , can, bottle, coke
	num_classes = 4

	
	# get the model using our helper function
	model = get_instance_segmentation_model(num_classes)

	# PRETRAINED Model File Check
	if downloader.filecheck('drinks_model_santiago.pth', 176220000):
		model.load_state_dict(torch.load('drinks_model_santiago.pth'))
	else:
		print('ERROR: MODEL FILE NOT FOUND')
		return None

	# move model to the right device
	model.to(device)

	return model


from PIL import Image, ImageDraw
import cv2
from torchvision.transforms import functional as F
import config
config.params

def show(model_cust, device):

	
	# define a video capture object
	stream = cv2.VideoCapture(0)
	
	# stream = cv2.VideoCapture('data/test1.mp4')

	try:
		while(True):
			
			# Capture the video frame
			# by frame
			ret, frame = stream.read()

			# Convert From CV2 to PIL
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			im_pil = Image.fromarray(img)

			# Convert from PIL to Tensor
			img_T = F.to_tensor(im_pil)

			
			with torch.no_grad():
				prediction = model_cust([img_T.to(device)])

			# breakdown prediction Results
			names = prediction[0]['labels']
			boxes = prediction[0]['boxes']
			scores = prediction[0]['scores']
			masks = prediction[0]['masks']

			im = Image.fromarray(img_T.mul(255).permute(1, 2, 0).byte().numpy())
			draw = ImageDraw.Draw(im)
			for idx, s in enumerate(scores):
				if s > 0.6:
					print(idx,s)

					# idx = 0
					tag = names[idx]
					bbox = boxes[idx].tolist()

					classname = config.params['classes'][tag]
					color = config.params['bbox_color'][tag]
					draw.rectangle(bbox, outline=color, width=4)

			#  Convert Layered PIL back to CV2
			open_cv_image = np.array(im).copy()
			open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
			cv2.imshow("Mask", open_cv_image)

			# Display the resulting frame
			cv2.imshow('frame', frame)
			
			# the 'q' button is set as the
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	except Exception as e:
		print('Interupt:', e)	

	# After the loop release the cap object
	stream.release()
	# Destroy all the windows
	cv2.destroyAllWindows()
	

if __name__ == '__main__':
	print('Detector Live')
	model_cust = main()

	# Set Model to Evalute MOde
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model_cust.eval()
	show(model_cust = model_cust, device = device)