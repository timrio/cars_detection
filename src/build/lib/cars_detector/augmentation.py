from skimage.filters import gaussian
import numpy as np
import cv2

class Transfo:
    def __init__(self, img):
        self.img = img.astype(np.uint8)

    def increase_brightess(self):
        self.img = np.clip(self.img*1.2, 0, 255).astype(np.uint8)

    def add_noise(self):
        row,col,ch = self.img.shape         
        #White
        pts_x = np.random.randint(0, col-1 , 100) #From 0(col-1)Make a thousand random numbers up to
        pts_y = np.random.randint(0, row-1 , 100)
        self.img[(pts_y,pts_x)] = (255,255,255) #y,Note that the order is x

        #black
        pts_x = np.random.randint(0, col-1 , 100)
        pts_y = np.random.randint(0, row-1 , 100)
        self.img[(pts_y,pts_x)] = (0,0,0)

    def distort(self):
        ###Distort the image
        self.img = cv2.flip(self.img, 1) #Flip horizontally
        self.img = cv2.flip(self.img, 0) #Invert vertically


    def blur(self):
        blured_img = 255*gaussian(self.img, 1)
        self.img = blured_img.astype(np.uint8)
        

def random_augmentation(img):
    new_img = Transfo(img)
    number_of_transfo = np.random.randint(1,3)
    transfo_list = [new_img.increase_brightess, new_img.blur, new_img.distort, new_img.add_noise]
    transfo_to_apply = np.random.choice(transfo_list,number_of_transfo,replace=False)
    for transfo in transfo_to_apply:
        transfo()
    return(new_img.img)