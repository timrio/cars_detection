import numpy as np
import cv2
from cars_detector.features import compute_features


def first_window_sliding(img, clf, scaler):
    """
    impement a first window sliding to generate region of interest proposals
    high recall, low precision
    """
    # List of bounding box positions
    boxes_list = []
    pred_array = np.zeros((img.shape[0], img.shape[1]))
    ystart_ystop_scale = [(100, 450, 0,1280,1.5), (100, 450, 0,1280,4),
     (100, 450, 0,1280,3),(100, 450, 0,1280,2)]
    step = 40
    proba = 0.1 # we want a high recall with this first classifier
    # Searching different size windows at different scales:
    for (ystart, ystop, xstart, xstop, scale) in ystart_ystop_scale:
        # Crop
        current_img = img[ystart:ystop, xstart:xstop, :]
        if scale != 1:
            current_img = cv2.resize(current_img, (np.int32(current_img.shape[1]/scale), np.int32(current_img.shape[0]/scale)))
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        x_size = current_img.shape[1]
        y_size = current_img.shape[0]
        current_step = np.int32(step/scale)
        for xb in range(0, x_size-64, current_step):
            for yb in range(0, y_size-64, current_step):
                    
                xleft = np.int32(xb*scale)
                ytop = np.int32(yb*scale)

                # Extract the image patch
                try:
                    crop = current_img[ytop:ytop+64, xleft:xleft+64,:]
                    if crop.shape[0]!=64 or crop.shape[1]!=64:
                        crop = cv2.resize(crop, (64,64))
                except:
                    continue
                # compute features
                current_features = compute_features(crop)
                # Scale features and make a prediction
                current_features = scaler.transform([current_features]) 
                current_proba = clf.predict_proba(current_features)[0][1]
                xbox_left = np.int32(xleft*scale)
                ytop_draw = np.int32(ytop*scale)
                win_draw_x = np.int32(64*scale) 
                win_draw_y = win_draw_x
                if ytop_draw+ystart+win_draw_y > 500:
                    win_draw_y = 500 - ytop_draw - ystart
                # Append Detection Position to list 
                if current_proba >= proba:
                    boxes_list.append([xbox_left+xstart,ytop_draw+ystart,win_draw_x,win_draw_y])
                    pred_array[ytop_draw+ystart:ytop_draw+ystart+win_draw_y, xbox_left+xstart:xbox_left+xstart+win_draw_x] += current_proba

    return boxes_list, pred_array


def window_sliding_advanced(img, boxes_list, clf, scaler):
    """
    implement a second an accurate window sliding on the previously sampled regions of interests
    high precision
    """
    # List of bounding box positions
    pred_array = np.zeros((img.shape[0], img.shape[1]))
    scales = [0.7,1,2,2.5,3,4,5]
    step = 10
    proba = 0.6 # we want a very good precision this time
    new_boxes = []
    for box in boxes_list:
        xstart, xstop = max(0,box[0]-32),min(box[0]+box[2]+32,img.shape[1]) # we add a small offset to look further
        ystart, ystop = max(200,box[1]-32),min(box[1]+box[3]+32,500)
        for scale in scales:
            current_img = img[ystart:ystop, xstart:xstop, :]
            if scale != 1:
                try:
                    current_img = cv2.resize(current_img, (np.int32(current_img.shape[1]/scale), np.int32(current_img.shape[0]/scale)))
                except:
                    continue
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
                        if crop.shape[0]!=64 or crop.shape[1]!=64:
                            crop = cv2.resize(crop, (64,64))
                    except:
                        continue
                    # compute features
                    current_features = compute_features(crop)
                    # Scale features and make a prediction
                    current_features = scaler.transform([current_features]) 
                    current_proba = clf.predict_proba(current_features)[0][1]
                    xbox_left = np.int32(xleft*scale)
                    ytop_draw = np.int32(ytop*scale)
                    win_draw = np.int32(64*scale)
                    pred_array[ytop_draw+ystart:ytop_draw+ystart+win_draw, xbox_left+xstart:xbox_left+xstart+win_draw] += current_proba
                    if current_proba > proba:
                        new_boxes.append([xbox_left+xstart,ytop_draw+ystart,win_draw,win_draw])         
    return new_boxes, pred_array