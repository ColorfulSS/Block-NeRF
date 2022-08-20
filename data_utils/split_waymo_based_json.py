# This code is used for spliting the waymo dataset into several block datasets
# based on the json got by split_block.py

# This will take up a lot of storage space,
# but if you are interested in the contents of each block,
# you can try to extract the data for each block

import glob
import os
import json
import numpy as np
import argparse
import shutil
from tqdm import tqdm


def get_hparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, default="train",
                        help='train or val to split')
    parser.add_argument('--split_type', type=str, default="all",
                        help='choose the index of the blocks which u want to split,like:block_0,block_1;'
                             'If u want to split all the blocks, use all')

    return vars(parser.parse_args())


if __name__ == "__main__":
    hparams=get_hparams()
    root_data_dir = "../data/WaymoDataset"
    output_data_dir = "../data/Split_Block"

    with open(os.path.join(root_data_dir, 'json/{0}.json'.format(hparams['split'])), 'r') as fp:
        img_meta = json.load(fp)

    with open(os.path.join(root_data_dir, 'json/split_block_{0}.json').format(hparams['split']), 'r') as fp:
        block_meta = json.load(fp)

    if hparams['split_type']=="all":
        for block in tqdm(block_meta):
            block_root_dir = os.path.join(output_data_dir, "{0}_{1}".format(block,hparams['split']), "images")
            os.makedirs(block_root_dir, exist_ok=True)
            os.makedirs(os.path.join(output_data_dir, block, "json"), exist_ok=True)
            meta_path = os.path.join(output_data_dir, block, f"json/train.json")
            split_meta = {}

            block_info = block_meta[block]
            count = 0
            for img_name in tqdm(block_info["elements"]):
                # 首先转移数据到对应文件夹中
                file_list = ['.png']  # , '_ray_dirs.npy', '_ray_origins.npy']
                for file in file_list:
                    old_file_path = os.path.join(root_data_dir, "images", img_name[0] + file)
                    # new_file_path = os.path.join(block_root_dir, img_name + file)
                    new_file_path = os.path.join(block_root_dir, str(count) + "_" + img_name[0] + file)
                    # 用于观察数据集连续情况

                    if os.path.exists(new_file_path):
                        print(f"{new_file_path} has been saved!")
                        continue
                    # 将文件从old_path复制到new_path
                    shutil.copyfile(old_file_path, new_file_path)
                count += 1
                # 提取相机的参数
                split_meta[img_name[0]] = img_meta[img_name[0]]
                with open(meta_path, "w") as fp:
                    json.dump(split_meta, fp)
                    fp.close()

            print(f"The {block}'s data has been saved in the path: {block_root_dir}")
            print()

    else:
        print("Begin to extract the images in {0}".format(hparams['split_type']))
        block=hparams['split_type']
        block_root_dir = os.path.join(output_data_dir, "{0}_{1}".format(block,hparams['split']), "images")
        os.makedirs(block_root_dir, exist_ok=True)
        os.makedirs(os.path.join(output_data_dir, block, "json"), exist_ok=True)
        meta_path = os.path.join(output_data_dir, block, f"json/train.json")
        split_meta = {}

        block_info = block_meta[block]
        count = 0
        for img_name in tqdm(block_info["elements"]):
            # 首先转移数据到对应文件夹中
            file_list = ['.png']  # , '_ray_dirs.npy', '_ray_origins.npy']
            for file in file_list:
                old_file_path = os.path.join(root_data_dir, "images", img_name[0] + file)
                # new_file_path = os.path.join(block_root_dir, img_name + file)
                new_file_path = os.path.join(block_root_dir, str(count) + "_" + img_name[0] + file)
                # 用于观察数据集连续情况

                if os.path.exists(new_file_path):
                    print(f"{new_file_path} has been saved!")
                    continue
                # 将文件从old_path复制到new_path
                shutil.copyfile(old_file_path, new_file_path)
            count += 1
            # 提取相机的参数
            split_meta[img_name[0]] = img_meta[img_name[0]]
            with open(meta_path, "w") as fp:
                json.dump(split_meta, fp)
                fp.close()

        print(f"The {block}'s data has been saved in the path: {block_root_dir}")
        print()
        #选择你想要提取的block
