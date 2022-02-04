import cv2

def get_interest_areas(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    new_rect = []
    for rect in rects:
        x = rect[1]
        y = rect[0]
        size_x = rect[3]
        size_y = rect[2]
        new_rect.append([x,y,size_x,size_y])
    return(new_rect)


def naive_window_sliding(img):
    boxes = []
    for size in range(100,200,20):
        for i in range(size, img.shape[0], int(size*0.75)):
            for j in range(size, img.shape[1], int(size*0.75)):
                boxes.append([i-size,j-size,size,size])
    return(boxes)


# pyramid window sliding

def pyramid(image, scale=1.5, minSize=(128, 128)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[0] / scale)
		h = int(image.shape[1] / scale)
		image = cv2.resize(image, (w,h))
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[0] or image.shape[0] < minSize[1]:
			break
		# yield the next image in the pyramid
		yield image