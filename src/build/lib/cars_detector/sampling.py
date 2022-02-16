import numpy as np
from cars_detector.utils import read_frame
from tqdm import tqdm
import cv2
from skimage.io import imread
import os
import random


def pyramid(image, scale=1.5, minSize=(128, 128)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[0] / scale)
		h = int(image.shape[1] / scale)
		image = cv2.resize(image, (h,w))
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[0] or image.shape[0] < minSize[1]:
			break
		# yield the next image in the pyramid
		yield image



def sampling_box_images(df_ground_truth, minSize=(64, 64), scale_step = 1.5):
    total_positive_samples = []
    total_negative_samples = []
    i = 0

    for frame_id in tqdm(range(1,df_ground_truth.shape[0]), position = 0):
        try:
            bbs = list(map(int, df_ground_truth.loc[frame_id].bounding_boxes.split(" ")))
            bbs = np.array_split(bbs, len(bbs) / 4)
        except:
            bbs = np.array([])
        img = read_frame(df_ground_truth, frame_id)
        for i,test_img in enumerate(pyramid(img, scale=scale_step, minSize=minSize)):
            scale = scale_step**i
            window = minSize[0]
            x_size = test_img.shape[1]
            y_size = test_img.shape[0]
            step = np.int32(40/scale)
            new_bbs = np.int32(np.array(bbs)/scale)
            mask_img = np.zeros((test_img.shape[0], test_img.shape[1]))
            for box in new_bbs:
                mask_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1

            for xb in range(0, x_size-window, step):
                for yb in range(0, y_size-window, step):
                    
                    xleft = np.int32(xb*scale)
                    ytop = np.int32(yb*scale)

                    crop = test_img[ytop:ytop+window,xleft:xleft+window,:]
                    
                    if crop.shape[0] == window and crop.shape[1] == window:
                        masked_crop = mask_img[ytop:ytop+window,xleft:xleft+window]
                        masked_crop_size = len(masked_crop.ravel())
                        if masked_crop_size>0:
                            if np.sum(masked_crop)/(masked_crop_size) > 0.5:
                                total_positive_samples.append(crop)
                            elif np.sum(masked_crop)/(masked_crop_size) < 0.05:
                                total_negative_samples.append(crop)
                            else:
                                continue
    return(total_positive_samples, total_negative_samples)

def get_vehicles_extra_images():
    positive_samples = []
    folders_list = os.listdir('vehicles/')
    for folder in folders_list:
        if folder != '.DS_Store':
            image_list = os.listdir('vehicles/'+folder)
            for image in image_list:
                try:
                    img = imread('vehicles/'+folder+'/'+image)
                    if img.shape[0] != 64 and img.shape[1] != 64:
                        img = cv2.reshape(img, (64,64,3))
                except:
                    continue
                positive_samples.append(img)
    return(positive_samples)

def get_non_vehicles_extra_images():
    negative_samples = []
    folders_list = os.listdir('non-vehicles/')
    for folder in folders_list:
        if folder != '.DS_Store':
            image_list = os.listdir('non-vehicles/'+folder)
            for image in image_list:
                try:
                    img = imread('non-vehicles/'+folder+'/'+image)
                    if img.shape[0] != 64 and img.shape[1] != 64:
                        img = cv2.reshape(img, (64,64,3))
                except:
                    continue
                negative_samples.append(img)
    return(negative_samples)


def sampling(df_ground_truth):
    total_positive_samples, total_negative_samples = sampling_box_images(df_ground_truth, minSize=(64, 64), scale_step = 1.5)

    extra_positive_samples = get_vehicles_extra_images()
    extra_negative_samples = get_non_vehicles_extra_images()

    total_negative_samples = random.sample(total_negative_samples, 15000)
    total_positive_samples = random.sample(total_positive_samples, 8000)

    total_positive_samples.extend(extra_positive_samples)
    total_negative_samples.extend(extra_negative_samples)

    return(total_positive_samples, total_negative_samples)



    
