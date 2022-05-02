import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class Res50Dataset(torch.utils.data.Dataset):
	def __init__(self, dictionary, root, transforms=None):
		self.dictionary = dictionary
		self.root = root
		self.transforms = transforms
		# load all image files, sorting them to
		# ensure that they are aligned
		# self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
		# self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

	def __getitem__(self, idx):
		key = list(self.dictionary.keys())[idx]
		d_val = self.dictionary[key]
		# print('k', key)
		# print('d',d_val)


		# load images ad masks
		img_path = d_val['filename']
		img = Image.open(img_path).convert("RGB")
		# note that we haven't converted the mask to RGB,
		# because each color corresponds to a different instance
		# with 0 being background

		masks = d_val['masks']
		# print('m',masks)
		# mask = Image.open(mask_path)
		# mask = np.array(mask)
		# instances are encoded as different colors
		# obj_ids = np.unique(mask)
		# first id is the background, so remove it
		# obj_ids = obj_ids[1:]

		# split the color-encoded mask into a set
		# of binary masks

		num_objs = len(masks)
		# get bounding box coordinates for each mask
		mask_list = []
		boxes = []
		areas = []
		labels = []
		for mask in masks:
			name = int(mask['Name'])
			mask_list.append(name)
			if name not in labels:
				labels.append(name)

			xmin = np.min(mask['x'])
			xmax = np.max(mask['x'])
			ymin = np.min(mask['y'])
			ymax = np.max(mask['y'])
			boxes.append([xmin, ymin, xmax, ymax])
			
			areas.append((ymax - ymin) * (xmax - xmin))

		# obj_ids = labels
		boxes = torch.as_tensor(boxes, dtype=torch.float32)

		# there is only one class
		labels = torch.as_tensor(labels, dtype=torch.int64)
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

		print('ZZZ:', mask_list)
		# masks = torch.as_tensor(masks, dtype=torch.uint8)

		id = key[key.find(self.root) + len(self.root) + 1:key.rfind('.jpg')]
		id = int(id)
		# print(id)
		image_id = torch.tensor([id])
		# suppose all instances are not crowd
		iscrowd = torch.zeros(labels.size(), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes		#each mask
		target["labels"] = labels	# Each bbox
		# target["masks"] = [masks]
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target

	def __len__(self):
		return len(self.dictionary)

