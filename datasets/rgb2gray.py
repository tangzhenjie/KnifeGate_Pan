
import cv2
import os
import numpy as np

class0 = np.array([0, 0, 0])
class1 = np.array([255, 255, 255])

def encode_labels(image_color_data):
    iamge_data_RGB = image_color_data
    height, width, chanel = iamge_data_RGB.shape
    label_seg = np.zeros([height, width], dtype=np.int8)
    label_seg[(iamge_data_RGB == class0).all(axis=2)] = 0
    label_seg[(iamge_data_RGB == class1).all(axis=2)] = 1

    return label_seg


# 把RGB标签换成灰度标签
if __name__ == "__main__":

    input_path = "./origin/labels/"

    output_path = "./origin/new_labels/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    for file_name in os.listdir(input_path):
        filname_path = input_path + file_name
        grb_data = cv2.imread(filname_path)
        gray_data = encode_labels(grb_data)
        cv2.imwrite(output_path + file_name, gray_data)
