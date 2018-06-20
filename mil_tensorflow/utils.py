import random
from scipy.stats import entropy as scipy_entropy
import cv2
import numpy as np
import pydicom
import os

def readUncomp(fname):
	'''
	Function to read decompressed dicom files and return header and volume
	Input: fname -  Full path to file to be read (string)
	Output: RefIds: Metadata
			pix: Volume (Numpy array: Z x X x Y) 
	'''
	RefDs = pydicom.read_file(fname)
	pix = RefDs.pixel_array
	return RefDs, pix

def readQCA(ds):
	'''
	Reads relevent clinical info from QCA file into a dictionary
	Input: ds: Metadata of the QCA dicom file to be read
	Output: header: structure containing relevant fields as lists, for each obstruction
	'''
	header = dict() #Dictionary to store relevant info
	tag = ds[0x08, 0x1140].value[0]
	header['SOPInstanceUID'] = tag[0x08,0x1155].value
	num = len(ds[0x7901,0x1099].value)
	
	header['num'] = num  #Number of obstructions. 
	header['Frame'] = []
	header['Obstruction Diameter'] = []
	header['Reference Diameter'] = []
	header['Diameter Stenosis'] = []
	header['Area Stenosis'] = []
	header['Obstruction Length'] = []
	header['Original,Upper'] = []
	header['Original,Lower'] = []
	header['Lesion,Upper'] = []
	header['Lesion,Lower'] = []
	header['Midline'] = []

	for i in range(num): #Iterate through each obstruction defined
		tags = ds[0x7901,0x1099].value[i]
		header['Frame'].append(tags[0x7903,0x1021].value + 1)
		tags = tags[0x7903, 0x1099].value[0] 
		header['Obstruction Diameter'].append(tags[0x7905, 0x1051].value) #in mm
		header['Lesion,Upper'].append(tags[0x7905,0x1071].value)
		header['Lesion,Lower'].append(tags[0x7905,0x1072].value)
		header['Original,Upper'].append(tags[0x7905,0x1074].value)
		header['Original,Lower'].append(tags[0x7905,0x1075].value)
		header['Midline'].append(tags[0x7905,0x107D].value)
		tags = tags[0x7905,0x1099].value[0]
		header['Reference Diameter'].append(tags[0x7909,0x1034].value)  #in mm
		header['Diameter Stenosis'].append(tags[0x7909,0x1035].value) #in %
		header['Area Stenosis'].append(tags[0x7909,0x1040].value)  #in %
		header['Obstruction Length'].append(tags[0x7909,0x1069].value)  #in mm

	return header



def is_vessel(patch):
	
	f = 0.4
	b = 15
	patch = cv2.resize(patch,None,fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
	# patch =  cv2.bilateralFilter(patch.astype(np.float32),9,75,75)
	patch = normalize_img(patch)
	rows,cols = patch.shape
	patch = patch[b:rows-b, b:cols-b]
	if np.var(patch)>0.003:
		value = 1
	else:
		value = 0
	return value

# def is_lesion(x_min,y_min, x_max,y_max, les_x_min, les_y_min, les_x_max, les_y_max):
# 	int_area = max(min(les_x_max,x_max) - max(les_x_min, x_min), 0) * max(min(les_y_max,y_max) - max(les_y_min, y_min), 0)
# 	les_area = (les_x_max - les_x_min) * (les_y_max - les_y_min)
# 	ratio = int_area/les_area
# 	if ratio>0.6:
# 	    value = 1
# 	else:
# 	    value = 0
# 	return value

def normalize_img(img):
	# clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
	# img = clahe.apply(img)
	img = img - np.min(img); 
	img = img/(np.max(img)+0.0001)
	return img

def check_patch_lesion(x_min, y_min, x_max, y_max, les_coord, frame):
	n = len(les_coord)
	j = -1
	for i in range(n):
		les_x_min = les_coord[i][0];les_y_min = les_coord[i][1];
		les_x_max = les_coord[i][2];les_y_max = les_coord[i][3];

		int_area = max(min(les_x_max,x_max) - max(les_x_min, x_min), 0) * max(min(les_y_max,y_max) - max(les_y_min, y_min), 0)
		les_area = (les_x_max - les_x_min) * (les_y_max - les_y_min)
		ratio = int_area/les_area
		if  ratio>0.7 and les_coord[i][5] == frame: #Higher value, as high false negative rate
			j = i
	if j>-1:
		value = les_coord[j][4]
	else:
		value = 0
	return value


def retrieve_img(name, fname, nearby = 0):
	a = name
	pnum = int(a.split('_')[1])
	file_no = int(a.split('_')[2])
	case = int(a.split('_')[3].split('.')[0])
	meta, vol = readUncomp(fname)
	qca_info = readQCA(meta)
	PATCH_SIZE = 128
	frames = []; imgs = []; les_coord = []
	num_lesions = qca_info['num']
	prev_imgs = []; next_imgs = []

	for i in range(num_lesions):
		frame = qca_info['Frame'][i]-1
		if vol.ndim == 3:
			img = vol[frame,:,:]
		else:
			img = vol
		orig_upper = qca_info['Original,Upper'][i]
		orig_lower = qca_info['Original,Lower'][i]
		xu = np.array(orig_upper[0::2]);yu = np.array(orig_upper[1::2]);
		xl = np.array(orig_lower[0::2]);yl = np.array(orig_lower[1::2]);
		x_min = np.amin(np.concatenate([xu,xl])); y_min = np.amin(np.concatenate([yu,yl]))
		x_max = np.amax(np.concatenate([xu,xl])); y_max = np.amax(np.concatenate([yu,yl]))
		les = np.zeros(shape = (6,))
		les[0] = x_min; les[1] = y_min; les[2] = x_max; les[3] = y_max; 
		les[4] = qca_info['Area Stenosis'][i] ; les[5] = frame
		les_coord.append(les)
		imgs.append(img)
		frames.append(frame)
		if nearby ==1:
			if vol.ndim == 3:
				if vol.shape[0]>frame+1:
					next_im = vol[frame+1,:,:]
				else:
					next_im = []
				if frame - 1>=0:
					prev_im = vol[frame-1,:,:]
				else:
					prev_im = []
			else:
				prev_im = []; next_im = []
			prev_imgs.append(prev_im)
			next_imgs.append(next_im)
	return pnum, file_no, frames, imgs, les_coord, prev_imgs, next_imgs



def get_lesion_extent(les_coord, frame):
	n = len(les_coord)
	for i in range(n):
		if les_coord[i][5] == frame:
			ind = i
	return (les_coord[ind][0],les_coord[ind][1],les_coord[ind][2],les_coord[ind][3], les_coord[ind][4])

