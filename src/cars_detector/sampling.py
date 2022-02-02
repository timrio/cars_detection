import numpy as np
from cars_detector.utils import read_frame, rgb2gray
from tqdm import tqdm
import cv2


def get_positive_sample_and_image(frame_id, df_ground_truth, thresh = 0.2):
    positive_samples = []
    try:
        bbs = list(map(int, df_ground_truth.loc[frame_id].bounding_boxes.split(" ")))
        bbs = np.array_split(bbs, len(bbs) / 4)
    except:
        bbs = np.array([])
    img = read_frame(df_ground_truth, frame_id)
    for x, y, dx, dy in bbs:
        positive_sample = img[y:y+dy,x:x+dx]
        positive_samples.append(positive_sample)

    # crop larger boxes
    for x, y, dx, dy in bbs:
        r = np.random.rand(1)
        if r <= thresh:
            try:
                additional_dx_up = np.random.randint(0,int(dx/2))
                additional_dx_down = np.random.randint(0,int(dx/2))
                additional_dy_up = np.random.randint(0,int(dy/2))
                additional_dy_down = np.random.randint(0,int(dy/2))
                positive_sample = img[max(0,y+additional_dy_down):min(y+dy+additional_dy_up, img.shape[0]),max(0,x+additional_dx_down):min(x+dx+additional_dx_up, img.shape[1])]
                positive_samples.append(positive_sample)
            except:
                continue

    # crop smaller boxes
    for x, y, dx, dy in bbs:
        r = np.random.rand(1)
        if r <= thresh:
            try:
                additional_dx_up = np.random.randint(0,int(dx/2))
                additional_dx_down = np.random.randint(0,dx-additional_dx_up)
                additional_dy_up = np.random.randint(0,int(dy/2))
                additional_dy_down = np.random.randint(0,dy-additional_dy_up)
                positive_sample = img[y+additional_dy_down:y+dy-additional_dy_up,x+additional_dx_down:x+dx-additional_dx_up]
                if positive_sample.shape[0] >= 50 and positive_sample.shape[1] >= 50:
                    positive_samples.append(positive_sample)
            except:
                continue
    return(positive_samples, bbs, img)



def get_negative_sample(img, positive_bbs, number_of_negative_samples):
    # get negative sample
    try:
        dx_list = [box[2] for box in positive_bbs]
        dy_list = [box[3] for box in positive_bbs]
        min_dx, max_dx = np.min(dx_list), np.max(dx_list)
        min_dy, max_dy = np.min(dy_list), np.max(dy_list)
    except:
        min_dx, max_dx = 64, 128
        min_dy, max_dy = 64, 128


    number_of_neg_examples = 0
    negative_samples = []
    negative_bbs = []
    gray_img = rgb2gray(img)
    for x, y, dx, dy in positive_bbs:
        gray_img[y:y+dy,x:x+dx] = 1000000

    i = 0
    while (number_of_neg_examples <= number_of_negative_samples) and (i <= number_of_negative_samples + 3000):
        i+=1
        try:
            dx = np.random.randint(min_dx,max_dx)
            dy = np.random.randint(max(min_dy, dx//2), min(max_dy,2*dx))
            x = np.random.randint(dx,img.shape[1]-dx)
            y = np.random.randint(dy,img.shape[0]-dy)
            crop_gray = gray_img[y:y+dy,x:x+dx]
            if np.max(crop_gray) <= 1000:
                crop = img[y:y+dy,x:x+dx]
                negative_samples.append(crop)
                negative_bbs.append([x,y,dx,dy])
                gray_img[y:y+dy,x:x+dx] = 1000000
            number_of_neg_examples = len(negative_samples)
        except:
            continue
    return(negative_samples, negative_bbs)


def sampling(df_ground_truth):
    total_positive_samples = []
    total_negative_samples = []
    for frame_id in tqdm(range(1,df_ground_truth.shape[0]), position = 0):
        positive_samples, positive_bbs, img = get_positive_sample_and_image(frame_id, df_ground_truth)
        negative_samples, negative_bbs = get_negative_sample(img, positive_bbs, len(positive_bbs)+20)
        for img in positive_samples:
            total_positive_samples.append(cv2.resize(img, (64,64)))
        for img in negative_samples:
            total_negative_samples.append(cv2.resize(img, (64,64)))
    return(total_positive_samples, total_negative_samples)