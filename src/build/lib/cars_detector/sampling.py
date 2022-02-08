import numpy as np
from cars_detector.utils import read_frame
from tqdm import tqdm
import cv2
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



def sampling(df_ground_truth, minSize=(100, 100), scale_step = 1.5):
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
            step = 35

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
                            if np.sum(masked_crop)/(masked_crop_size) > 0.9:
                                total_positive_samples.append(crop)
                            else:
                                total_negative_samples.append(crop)
    return(total_positive_samples, total_negative_samples)