import cv2
import numpy as np
import os
from PIL import Image


in_dir = "./result/pre/"
out_dir = "./result/postprocessor_pre/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file_name in os.listdir(in_dir):
    file_path = in_dir + file_name

    # read gray image
    img_orign = cv2.imread(file_path, 0)

    # erode and dilate the image
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_orign, kernel, iterations=5)
    img_erode = cv2.erode(img_dilate, kernel, iterations=4)
    cv2.imwrite(out_dir + file_name, img_erode)


# file_path = "./result/pre/pre11.jpg"
#
# # read gray image
# img_orign = cv2.imread(file_path, 0)
#
# # erode and dilate the image
# kernel = np.ones((5, 5), np.uint8)
# img_erode = cv2.erode(img_orign, kernel, iterations=2)
# cv2.imshow("erode", np.hstack((img_orign, img_erode)))
#
# img_dilate = cv2.dilate(img_erode, kernel, iterations=2)
#
# cv2.imshow("dilate", np.hstack((img_erode, img_dilate)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()



