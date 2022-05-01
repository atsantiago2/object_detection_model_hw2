# Downloader based on this thread: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
# Some parts were rewritten to accomodate large data download prompt
# File Extraction based on: https://stackoverflow.com/questions/30887979/i-want-to-create-a-script-for-unzip-tar-gz-file-via-python

import os

import requests
import tarfile


def download_file_from_google_drive(id, destination):
	URL = "https://drive.google.com/uc?id=" + id + "&export=download&confirm=t"
	session = requests.Session()
	response = session.get(URL, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	save_response_content(response, destination)    

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def check_create_directory(dpath = 'data'):
	data_dir = os.path.isdir(dpath)
	if not data_dir:
		os.makedirs(dpath)
		print('Directory: {} - created'.format(dpath))
	else:
		print('Directory: {} - exists'.format(dpath))

def filecheck(fpath, size):
	if os.path.exists(fpath):
		fsize = os.path.getsize(fpath)
		print("File: \t\t{}\nFile Size:\t{}".format(fpath, fsize))
		if fsize >= size:
			return True
		else:
			return False
	else:
		return False


def get_drinks_dataset(filename = 'drinks.tar.gz', filesize = 146820533):
	file_id = '1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'

	# Check Data Storage Location
	destination = os.path.join('data', '')
	check_create_directory(destination)

	# Check Data Set File Status
	destination = os.path.join('data', 'drinks.tar.gz')
	if filecheck(destination, 146820000):
		print("Data Set Already Downloaded")
	else:
		print("Downloading Data Set")
		download_file_from_google_drive(file_id, destination)

	# Check if Extracted Data Set Exist
	# If Drinks Folder EXISTS: Assume Extraction Already Completed
	destination_directory = os.path.join('data', 'drinks')
	if not os.path.isdir(destination_directory):
		print("Extracting Drinks to Folder")

		#Change Path to data folder for extract all
		destination_directory = os.path.join('data')
		if destination.endswith("tar.gz"):
			tar = tarfile.open(destination, "r:gz")
			tar.extractall(destination_directory)
			tar.close()
			print("Drinks Data Set Extracted")
			return True
		elif destination.endswith("tar"):
			tar = tarfile.open(destination, "r:")
			tar.extractall(destination_directory)
			tar.close()
			print("Drinks Data Set Extracted")
			return True
		else:
			print("Drinks Extraction Failed")
			return False
	else:
		print("Drinks Folder Already Exists")
		return True



	



if __name__ == "__main__":
	file_id = id
	get_drinks_dataset()
	# download_file_from_google_drive(file_id, destination)
	# download_file_from_google_drive_confirmed_noscan(file_id, destination)