from skimage import exposure
from skimage import feature
from tqdm import tqdm


def compute_features(total_positive_samples,total_negative_samples):
    positive_features = []
    negative_features = []
    for img in tqdm(total_positive_samples, position = 0):
        (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        positive_features.append(H)
    for img in tqdm(total_negative_samples, position = 0):
        (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        negative_features.append(H)
    return(positive_features, negative_features)