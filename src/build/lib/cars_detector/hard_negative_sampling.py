import numpy as np
from cars_detector.utils import read_frame
from tqdm import tqdm
import cv2
from skimage.io import imread
from cars_detector.features import compute_features


def hard_negative_sampling(df_ground_truth, clf, scaler):
    hard_negative_samples = []
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
                if np.sum(crop_mask)==0: # if there is obviously no car on the frame ...
                    crop = img[y:y+64, x:x+64,:]
                    features = compute_features(crop)
                    features = scaler.transform([features])
                    proba = clf.predict_proba(features)[0][1]
                    if proba >= 0.7: # ...but if the classifier truly thinks there is a car
                        hard_negative_samples.append(crop)
                else:
                    continue
    return(hard_negative_samples)







    
