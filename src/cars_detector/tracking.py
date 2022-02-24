import numpy as np
import pandas as pd



def tracking(image_index, mask_dict, boxes_array, search_radius = 1, number_of_frames_to_look = 15, freq_of_presence = 0.6):
    '''
    implement pseudo tracking to remove false positive examples
    if a box is not present enough close to the studied box in the past frames we consider it as a false negative example
    '''
    boxes_to_keep = []
    boxes = boxes_array.query('image==@image_index').boxes.values[0]

    for box in boxes:
        count = 0
        number_of_remaining_frames = min(number_of_frames_to_look, 203-image_index)
        min_number_of_presence = np.int32(number_of_remaining_frames*freq_of_presence)
        for j in range(1,number_of_remaining_frames): # if not enough remaining images above
            future_idx = image_index+j
            mask_future = mask_dict[future_idx]
            y_min, y_max = max(0, box[1]-j*search_radius), min(mask_future.shape[0], box[1]+box[3]+j*search_radius)
            x_min, x_max = max(0, box[0]-j*search_radius), min(mask_future.shape[1], box[0]+box[2]+j*search_radius)
            crop = mask_future[y_min:y_max,x_min:x_max]
            if box[0] < 300 and box[1] < 300: # nothing should be in this area
                continue
            if np.sum(crop) < 1000: # almost no other box has been found in past frames
                continue
            else:
                count += 1

            if count >= min_number_of_presence: # if the box has been found in a sufficient amount of past frames
                boxes_to_keep.append(box)
                break

    return(boxes_to_keep)