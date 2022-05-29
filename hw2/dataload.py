
import json, os
def json_get_segment_train_details(train):
	if train is True:
		a_fp = os.path.join('data','drinks', 'segmentation_train.json')
	else:
		a_fp = os.path.join('data','drinks', 'segmentation_test.json')

	with open(a_fp, 'r') as f:
		d1 = json.load(f)
		d1 = d1["_via_img_metadata"]

	return d1

def get_dataset_dict(train):
	d1 = json_get_segment_train_details(train=train)

	target = {}
	unique_classes = []
	for key, val in d1.items():
		# print(key,'val:', value)
		filename = os.path.join('data','drinks', val['filename'])

		# print(val['regions'][0])

		mask_list = []
		for mask in val['regions']:
			mask_d = {}
			# print(mask['shape_attributes'])
			# print(mask['region_attributes']['Name'])
			name = mask['region_attributes']['Name']
			mask_d['Name'] = name
			if name not in unique_classes:
				unique_classes.append(name)

			mask_d['x'] = mask['shape_attributes']['all_points_x']
			mask_d['y'] = mask['shape_attributes']['all_points_y']
			mask_list.append(mask_d)

		d_temp = {}
		d_temp['filename'] = filename
		d_temp['masks'] = mask_list
		# d_temp[items] = 
		# target[filename] 
		target[filename] = d_temp

	# print('Total Unique Masks', len(unique_classes))
	# print('Unique Masks', unique_classes)
	# target = d1
	return target, unique_classes