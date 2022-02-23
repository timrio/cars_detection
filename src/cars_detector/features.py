from skimage import exposure
from skimage import feature
from tqdm import tqdm
import numpy as np

def compute_hogs_features(img):
    H1 = feature.hog(img[:,:,1], orientations=9, pixels_per_cell=(16,16), cells_per_block=(3, 3))
    H2 = feature.hog(img[:,:,1], orientations=9, pixels_per_cell=(16,16), cells_per_block=(3, 3))
    H3 = feature.hog(img[:,:,1], orientations=9, pixels_per_cell=(16,16), cells_per_block=(3, 3))
    hog_features = np.hstack((H1,H2,H3))
    return(hog_features)


def compute_colors_features(img, nbins=32):   
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def compute_features(img):
    hogs_features = compute_hogs_features(img)
    color_features = compute_colors_features(img)
    features = np.concatenate((hogs_features, color_features))
    return(features)