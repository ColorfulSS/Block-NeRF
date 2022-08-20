import os
import json
import numpy as np
import argparse

# 提取每一个相机的路径对应的图像信息

def get_hparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default="../data/WaymoDataset", help='Where the json is')
    parser.add_argument('--save_dir', type=str,
                        default="../data/WaymoDataset", help='Where the json is')
    return vars(parser.parse_args())

def extract_cam_idx(train_meta):
    cam_idx=[]
    for meta in train_meta:
        img_info=train_meta[meta]
        if img_info['cam_idx'] not in cam_idx:
            cam_idx.append(img_info['cam_idx'])
    return sorted(cam_idx)

def extract_img_each_idx(idx,train_meta,train_split_meta):
    imgs=[]
    for block in train_split_meta:
        for element in train_split_meta[block]['elements']:
            if train_meta[element[0]]['cam_idx']==idx:
                if element[0] not in imgs:
                    imgs.append(element[0])
    return imgs



if __name__=="__main__":
    hparams=get_hparams()
    with open(os.path.join(hparams['root_dir'], 'json/train.json'), 'r') as fp:
        train_meta = json.load(fp)
    with open(os.path.join(hparams['root_dir'], 'json/split_block_train.json'), 'r') as fp:
        train_split_meta = json.load(fp)
    #1. 提取所有的相机idx
    cam_idxes=extract_cam_idx(train_meta)
    print(f"Totally there are {len(cam_idxes)} cameras. ")

    cam_save_path = os.path.join(hparams['save_dir'], "json/cam_info.json")

    #2. 根据split_train的结果按顺序提取每一个cam的拍摄路径
    cam_imgs={}
    for idx in cam_idxes:
        cam_imgs[idx]=extract_img_each_idx(idx,train_meta,train_split_meta)
        with open(cam_save_path, "w") as fp:
            json.dump(cam_imgs, fp)
            fp.close()

    print("The camera information has been saved in the path of {0}".format(cam_save_path))



