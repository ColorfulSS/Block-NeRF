import os
import json
import argparse
import shutil
from tqdm import tqdm

#   基于分割出来的json提取每一个相机的路径

def get_hparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--split_type', type=str, default="all",
                        help='choose the index of the blocks which u want to split,like:block_0,block_1;'
                             'If u want to split all the blocks, use all')

    return vars(parser.parse_args())

if __name__=="__main__":
    hparams=get_hparams()
    root_data_dir = "../data/WaymoDataset"
    output_data_dir = "../data/Split_cam"

    with open(os.path.join(root_data_dir, 'json/cam_info.json'), 'r') as fp:
        cam_meta = json.load(fp)

    if hparams['split_type']=="all":
        for cam in tqdm(cam_meta):
            cam_dir = os.path.join(output_data_dir, "{0}".format(cam))
            os.makedirs(cam_dir, exist_ok=True)
            split_meta = {}

            cam_list = cam_meta[cam]
            count = 0
            for img_name in tqdm(cam_list):
                # 首先转移数据到对应文件夹中
                file_list = ['.png']  # , '_ray_dirs.npy', '_ray_origins.npy']
                for file in file_list:
                    old_file_path = os.path.join(root_data_dir, "images", img_name + file)
                    # new_file_path = os.path.join(block_root_dir, img_name + file)
                    new_file_path = os.path.join(cam_dir, str(count) + "_" + img_name + file)
                    # 用于观察数据集连续情况

                    if os.path.exists(new_file_path):
                        print(f"{new_file_path} has been saved!")
                        continue
                    # 将文件从old_path复制到new_path
                    shutil.copyfile(old_file_path, new_file_path)
                count += 1
                # 提取相机的参数
                
            print(f"The camera {cam}'s data has been saved in the path: {cam_dir}")
            print()