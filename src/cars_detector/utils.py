from skimage.io import imread
import pandas as pd
import os


def rgb2gray(rgb):
    """
    convert image to gray
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def read_frame(df_train, frame):
    """Read frames and create integer frame_id-s"""
    file_path = df_train[df_train.index == frame]['frame_id'].values[0]
    return imread(file_path)


def read_test_frame(frame):
    """Read frames and create integer frame_id-s"""
    format_id = f"{frame:03d}"
    return imread(os.path.join('./test/', format_id+'.jpg'))
