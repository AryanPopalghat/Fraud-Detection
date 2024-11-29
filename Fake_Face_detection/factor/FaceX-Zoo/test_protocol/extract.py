import os
import numpy as np
from extract_feature import extract_features

def extract_features_from_folder(folder_path, save_folder):
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        features = extract_features(image_path)
        feature_path = os.path.join(save_folder, os.path.splitext(image_name)[0] + '.npy')
        np.save(feature_path, features)
