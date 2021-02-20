# import os
# import shutil
# path = "./train/images/"
# despath = "./train/labels/"
#
# for name in os.listdir(path):
#     if name.endswith(".json"):
#         shutil.move(path+name, despath)
# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np
import os
import shutil


def cvt_one(json_path, img_path, save_path, label_color):
    # load img and json
    data = json.load(open(json_path,encoding='gbk'))
    img = cv2.imread(img_path)

    # get background data
    img_h = data['imageHeight']
    img_w = data['imageWidth']
    color_bg = (0, 0, 0)
    points_bg = [(0, 0), (0, img_h), (img_w, img_h), (img_w, 0)]
    img = cv2.fillPoly(img, [np.array(points_bg)], color_bg)

    # draw roi
    for i in range(len(data['shapes'])):
        name = data['shapes'][i]['label']
        points = data['shapes'][i]['points']
        #color =  data['shapes'][i]['fill_color']
        # data['shapes'][i]['fill_color'] = label_color[name]  # 修改json文件中的填充颜色为我们设定的颜色
        if label_color:
            img = cv2.fillPoly(img, [np.array(points, dtype=int)], label_color[name])
        else:
            print("Error")
            #img = cv2.fillPoly(img, [np.array(points, dtype=int)], (color[0], color[1], color[2]))
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    save_dir ='./origin/labels'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_dir = './origin'
    files = os.listdir(file_dir)
    img_files = list(filter(lambda x: '.jpg' in x, files))
    label_color = { #设定label染色情况
        'kg_close': (255, 255, 255)
    }
    save_img='./origin/imgs'
    if not os.path.exists(save_img):
        os.makedirs(save_img)
    for i in range(len(img_files)):
        img_path = file_dir + '/' + img_files[i]
        shutil.copy(img_path, save_img + '/' + img_files[i])
        json_path = img_path.replace('.jpg', '.json')
        print(json_path)
        save_path = save_dir + '/' + img_files[i]
        print('Processing {}'.format(img_path))
        cvt_one(json_path, img_path, save_path, label_color)

