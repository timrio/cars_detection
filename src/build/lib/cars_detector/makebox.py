import numpy as np
import cv2

def non_max_suppression(boxes, overlapThresh = 0.4):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 1]  # x coordinate of the top-left corner
    y1 = boxes[:, 0]  # y coordinate of the top-left corner
    x2 = x1+boxes[:, 3]  # x coordinate of the bottom-right corner
    y2 = y1+boxes[:, 2]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[1], boxes[temp_indices,1])
        yy1 = np.maximum(box[0], boxes[temp_indices,0])
        xx2 = np.minimum(box[1]+box[3], boxes[temp_indices,1]+boxes[temp_indices,3])
        yy2 = np.minimum(box[0]+box[2], boxes[temp_indices,0]+boxes[temp_indices,2])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    #return only the boxes at the remaining indices
    return boxes[indices].astype(int)


def box_otsu(pred_array):
	gray = pred_array.astype("uint8")
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

	# Find contours
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	bbs = []
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		bbs.append([x,y,w,h])
	return(bbs)

