from glob import glob
import matplotlib.pyplot as plt
import os

import argparse

from numpy import character
import cv2

class img_preprocess:
    def __init__(self, img_path):
        self.img_path = img_path
        
    def create_list(self):
        self.img_list = glob(img_path + '/*.png')
        print('Here is {} images'.format(len(self.img_list)))
        self.set_images_label()
        
    def set_images_label(self):
        self.imgs = []
        self.labels = []
        self.max_length = 0
        
        for imgpath in self.img_list:
            self.imgs.append(imgpath)
            label = os.path.splitext(os.path.basename(imgpath))[0]
            self.labels.append(label)
            
            if len(label) > self.max_length:
                max_length = len(label)

        print(len(self.imgs), len(self.labels), self.max_length)
    
    def encoding(self):
        characters = set(''.join(self.labels))
        
        self.char_to_num = {}
        self.num_to_char = {}
        
        for idx, item in enumerate(characters):
            self.char_to_num[idx] = item
            self.num_to_char[item] = idx
        
        encoded = self.char_to_num( item for item in self.labels[0])
        print(encoded)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, help='enter your img path')

    opt = parser.parse_args()
    img_path = opt.img_path

    preprocess = img_preprocess(img_path=img_path)
    preprocess.create_list()
    preprocess.encoding()