#Download Dataset

import downloader
downloader.get_drinks_dataset()

#Download Pre Trained Model

import os
#Get Model from Git
pret_model_url = 'https://github.com/atsantiago2/dl_course_proj2.git'
pret_model_dirname = 'git_pretrained_models'
pret_model_filename = 'testfile.txt'
pret_model_dirpath = os.path.join(pret_model_dirname, '')
pret_model_path = os.path.join(pret_model_dirname, pret_model_filename)

#Download Model from Online
from git import Repo
if os.path.isdir(pret_model_dirpath):
	print('Pretrained Model Already Downloaded Before')
else:
	Repo.clone_from(pret_model_url, pret_model_dirpath)
	print('Pretrained Model Downloaded Already')


#Copy Model to local
import shutil
if os.path.isfile(pret_model_filename):
	print('Pretrained Model Already in Place')
else:
	shutil.copyfile(pret_model_filepath, pret_model_filename)
	print('Pretrained Model Copy Complete')

#Train using GPU
#Show Performance Metrics