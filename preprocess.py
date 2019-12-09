# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:41:37 2019

@author: Jimmy Hua
"""

import numpy as np
from tqdm import tqdm
import cv2
import random
import os

np.random.seed(2019)
random.seed(2019)

def aug_img(img_path, flip=1, is_bright=1):
    img = cv2.imread(img_path)
    image = img.astype(np.float32)

    if flip:
        image = image[:, ::-1]

    expand_image = np.zeros((300,150, 3), dtype=image.dtype)
    expand_image[:,:,:] = 128
    left = random.uniform(0, 22)
    top = random.uniform(0, 34)
            
    br = random.uniform(1.3,1.6)
    bk = random.uniform(0.6, 0.8)
    if is_bright:
        image *= br
    else:
        image *= bk
    
    # 限制image的范围[0, 255.0]
    image = np.clip(image, 0, 255)
    expand_image[int(top):int(top+256), int(left):int(left+128)] = image
    #            
    #img3 = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img3 = expand_image.astype(np.uint8)
    x1 = int(round(random.uniform(0, 22)))
    y1 = int(round(random.uniform(0, 34)))
    
    img_crop = img3[y1:y1+256, x1:x1+128]
    return img_crop

def aug_data(name, base, f, b, train_txt, new_dir='crop'):
    img_path = base + name
    img_crop = aug_img(img_path, flip=f, is_bright=b)
    crop_name = name.replace('train', new_dir).split('.')[0] + ('_crop_%d_%d.jpg')%(f, b)
    crop_path = base + crop_name
    cv2.imwrite(crop_path, img_crop)
    with open(train_txt, 'a') as f1:
        f1.write(crop_name + ' '+ pid +'\n')   

base = './'

with open('train_list.txt', 'r') as f:
    lines = f.readlines()

data = {} 
for line in lines:
    _, people = line.strip().split()

    if people not in data.keys():
        data[people] = 1
    else:
        data[people] += 1

print('data is ready to aug....')

cout = 0

train_txt = 'aug_train_list.txt'

if os.path.exists(train_txt):
    os.remove(train_txt)

if not os.path.exists('crop'):
    os.makedirs('crop')

for line in (lines):

    name, pid = line.strip().split()
    img_path = base + name

    # 原图路径写入txt
    with open(train_txt, 'a') as f:
       f.write(name + ' '+ pid +'\n')
    
    # id 对应一张图的数据，增加4倍
    if data[pid] == 1:
        for f1 in range(2):
            for b1 in range(2):
                aug_data(name, base, f1, b1, train_txt)   
                cout += 1
                
    # id 对应两张图的数据，增加2倍
    elif data[pid] == 2:
        for b2 in range(2):
            aug_data(name, base, 1, b2, train_txt)
            cout += 1
            
    # id 对应三张图的数据，增加一倍
    elif data[pid] == 3:
        b3 = 1
        aug_data(name, base, 1, b3, train_txt)
        cout += 1
    print(cout)