import numpy as np
import cv2
from cars_detector.features import compute_features

# Extracts features using hog sub-sampling and make predictions
def first_window_sliding(img, clf, scaler):
    # List of bounding box positions
    boxes_list = []
    pred_array = np.zeros((img.shape[0], img.shape[1]))
    ystart_ystop_scale = [(200, 450, 1),(200, 450, 2),(200, 450, 4)]
    step = 64
    proba = 0.2 # we want a high recall with this first classifier
    # Searching different size windows at different scales:
    for (ystart, ystop, scale) in ystart_ystop_scale:
        box_vis=[]
        # Crop
        current_img = img[ystart:ystop, :, :]
        if scale != 1:
            imshape = current_img.shape
            current_img = cv2.resize(current_img, (np.int32(imshape[1]/scale), np.int32(imshape[0]/scale)))


        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        x_size = current_img.shape[1]
        y_size = current_img.shape[0]
        current_step = np.int32(step/scale)
        for xb in range(0, x_size-64, current_step ):
            for yb in range(0, y_size-64, current_step ):
                    
                xleft = np.int32(xb*scale)
                ytop = np.int32(yb*scale)

                # Extract the image patch
                try:
                    crop = current_img[ytop:ytop+64, xleft:xleft+64,:]
                except:
                    continue
                # compute features
             

                current_features = compute_features(crop)
                # Scale features and make a prediction
                current_features = scaler.transform([current_features]) 

                current_proba = clf.predict_proba(current_features)[1]
            
                if current_proba >= proba:
                    xbox_left = np.int32(xleft*scale)
                    ytop_draw = np.int32(ytop*scale)
                    win_draw = np.int32(64*scale)
                    # Append Detection Position to list 
                    boxes_list.append([xbox_left,ytop_draw+ystart,win_draw,win_draw])
                    pred_array[ytop_draw+ystart:ytop_draw+ystart+win_draw, xbox_left:xbox_left+win_draw] += 1

    return boxes_list, pred_array


def window_sliding_advanced(img, boxes_list, clf, scaler):
    # List of bounding box positions
    boxes_list = []
    pred_array = np.zeros((img.shape[0], img.shape[1]))
    scales = [0.5,0.7,1,1.3,1.5,1.7]
    step = 10
    proba = 0.95 # we want a very good precision this time
    for box in boxes_list:
        xstart, xstop = box[0],box[0]+box[2]
        ystart, ystop = box[1],box[1]+box[3]
        for scale in scales:
            current_img = img[ystart:ystop, xstart:xstop, :]
            if scale != 1:
                imshape = current_img.shape
                current_img = cv2.resize(current_img, (np.int32(imshape[1]/scale), np.int32(imshape[0]/scale)))
            x_size = current_img.shape[1]
            y_size = current_img.shape[0]
            current_step = np.int32(step/scale)
            for xb in range(0, x_size-64, current_step ):
                for yb in range(0, y_size-64, current_step ):
                    xleft = np.int32(xb*scale)
                    ytop = np.int32(yb*scale)
                    # Extract the image patch
                    try:
                        crop = current_img[ytop:ytop+64, xleft:xleft+64,:]
                    except:
                        continue
                    # compute features
                    current_features = compute_features(crop)
                    # Scale features and make a prediction
                    current_features = scaler.transform([current_features]) 
                    current_proba = clf.predict_proba(current_features)[1]
                    if current_proba >= proba:
                        xbox_left = np.int32(xleft*scale)
                        ytop_draw = np.int32(ytop*scale)
                        win_draw = np.int32(64*scale)
                        # Append Detection Position to list 
                        boxes_list.append([xbox_left+xstart,ytop_draw+ystart,win_draw,win_draw])
                        pred_array[ytop_draw+ystart:ytop_draw+ystart+win_draw, xbox_left+xstart:xbox_left+xstart+win_draw] += 1
    return boxes_list, pred_array
