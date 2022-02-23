import numpy as np
from cars_detector.utils import read_frame
from tqdm import tqdm
import cv2
from skimage.io import imread
import os
import random


def positive_sampling(df_ground_truth):
    positive_samples = []
    for frame_id in tqdm(range(1,df_ground_truth.shape[0]), position = 0):
        try:
            bbs = list(map(int, df_ground_truth.loc[frame_id].bounding_boxes.split(" ")))
            bbs = np.array_split(bbs, len(bbs) / 4)
        except:
            bbs = np.array([])
        img = read_frame(df_ground_truth, frame_id)
        for box in bbs:
            crop = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2],:]
            crop = cv2.resize(crop, (64,64))
            positive_samples.append(crop)
    return(positive_samples)




def negative_sampling(df_ground_truth):
    negative_samples = []
    for frame_id in tqdm(range(1,df_ground_truth.shape[0]), position = 0):
        try:
            bbs = list(map(int, df_ground_truth.loc[frame_id].bounding_boxes.split(" ")))
            bbs = np.array_split(bbs, len(bbs) / 4)
        except:
            bbs = np.array([])
        img = read_frame(df_ground_truth, frame_id)
        mask_img = np.zeros((img.shape[0], img.shape[1]))
        for box in bbs:
            mask_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]+=1
        for x in range(0, img.shape[1]-64, 64):
            for y in range(0, img.shape[0]-64,64):
                crop_mask = mask_img[y:y+64, x:x+64]
                if np.sum(crop_mask)!=0:
                    continue
                else:
                    crop = img[y:y+64, x:x+64,:]
                    negative_samples.append(crop)
    return(negative_samples)



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
    total_positive_samples = positive_sampling(df_ground_truth)
    total_negative_samples = negative_sampling(df_ground_truth)

    extra_positive_samples = get_vehicles_extra_images()
    extra_negative_samples = get_non_vehicles_extra_images()

    total_positive_samples.extend(extra_positive_samples)
    total_negative_samples.extend(extra_negative_samples)
    
    return(total_positive_samples, total_negative_samples)



    
