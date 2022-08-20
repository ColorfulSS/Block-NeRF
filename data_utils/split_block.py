import glob
import os
import json
import numpy as np
import argparse
from collections import defaultdict
import open3d as o3d
import random


def get_hparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--radius', type=float, default=0.3,
                        help='The radius of a block')
    parser.add_argument('--overlap', type=float,
                        default=0.5, help='overlap each block')
    parser.add_argument('--visualization', type=bool, default=True,
                        help="Whether visualize the split results")
    parser.add_argument('--visual_Block', type=bool, default=False,
                        help="When visualize whether visualize the split result")

    return vars(parser.parse_args())


def extract_imgname_origins(meta):
    img_origins = {}

    for idx, img_name in enumerate(meta):
        img_info = meta[img_name]
        origin = img_info['origin_pos']
        img_origins[img_name] = origin
    return img_origins


def resort_origins(img_origins, positions):
    # 调换一下img_origins的key和value
    origin2name = {}
    for img_orin in img_origins:
        origin = img_origins[img_orin]
        origin2name[tuple(origin)] = img_orin  # list不能做dict的key

    sorted_origins = {}
    for pos in positions:
        sorted_origins[origin2name[tuple(pos)]] = pos

    return sorted_origins


def get_the_distance(r=2, overlap=0.5):  # 根据半径和重叠率计算距离
    x = r * 0.9
    x0 = x
    # f用来描述方程的值，fd用来描述方程求导之后的值
    f = 2 * np.arccos(x0 / r) * (r ** 2) - 2 * x0 * \
        np.sqrt(r ** 2 - x0 ** 2) - overlap * np.pi * r ** 2
    fd = (2 * x0 ** 2 - 2 * r ** 2) / np.sqrt(r **
                                              2 - x0 ** 2) - 2 * np.sqrt(r ** 2 - x0 ** 2)
    h = f / fd
    x = x0 - h
    # 求更接近方程根的x的值
    while abs(x - x0) >= 1e-6:
        x0 = x
        # f用来描述方程的值，fd用来描述方程求导之后的值
        f = 2 * np.arccos(x0 / r) * (r ** 2) - 2 * x0 * \
            np.sqrt(r ** 2 - x0 ** 2) - overlap * np.pi * r ** 2
        fd = (2 * x0 ** 2 - 2 * r ** 2) / np.sqrt(r **
                                                  2 - x0 ** 2) - 2 * np.sqrt(r ** 2 - x0 ** 2)
        h = f / fd
        x = x0 - h

    return 2 * x


def get_each_block_element_train(img_train_origins, centroid, radius):
    # origins都是resort过的
    block_train_element = []

    index = 0
    for img_origin in img_train_origins:
        if np.linalg.norm(img_train_origins[centroid] - img_train_origins[img_origin]) <= radius:
            img_element = [img_origin, index]
            block_train_element.append(img_element)
            index += 1

    return block_train_element


def extract_img_base_camidx(cam_idx, train_meta):
    img_name = []
    for meta in train_meta:
        img_info = train_meta[meta]
        if img_info['cam_idx'] == cam_idx:
            img_name.append(meta)
    return img_name


def get_block_idx(img_name, split_train_results):
    for block in split_train_results:
        elements = split_train_results[block]["elements"]
        for element in elements:
            if element[0] == img_name:
                return [block, element[1]]
    return None


def get_val_block_index(img_val_origins, train_meta, val_meta, img_train_origins, split_train_results):
    split_val_results = defaultdict(list)
    for origin in img_val_origins:
        # 首先找到其对应的相机
        img_info = val_meta[origin]
        cam_idx = img_info['cam_idx']
        # 提取train_meta中所有相同idx的数据
        img_list = extract_img_base_camidx(cam_idx, train_meta)
        # 计算与其距离最近的img
        distance = 1000
        img_nearest = None
        for img in img_list:
            distance_temp = np.linalg.norm(img_val_origins[origin] - img_train_origins[img])
            if distance_temp < distance:
                distance = distance_temp
                img_nearest = img
        #print(img_nearest)
        block_name, index = get_block_idx(img_nearest, split_train_results)
        '''
        block_info = {
            'elements': block_train_element,
            "centroid": [centroid, img_train_origins_resort[centroid].tolist()]
        }
        split_train_results[f'block_{idx}'] = block_info
        '''
        # 得到最近的img_name,并找到其对应的block和对应的index

        split_val_results[block_name].append([origin, index])

    return split_val_results


def split_dataset(train_meta, val_meta, radius=0.5, overlap=0.5):
    img_train_origins = extract_imgname_origins(train_meta)
    img_val_origins = extract_imgname_origins(val_meta)

    train_positions = np.array([value for value in img_train_origins.values()])
    val_positions = np.array([value for value in img_val_origins.values()])

    train_indices = np.argsort(train_positions[:, 1])
    val_indices = np.argsort(val_positions[:, 1])

    train_positions = train_positions[train_indices, :]
    val_positions = val_positions[val_indices, :]

    img_train_origins_resort = resort_origins(img_train_origins, train_positions)
    img_val_origins_resort = resort_origins(img_val_origins, val_positions)

    distance = get_the_distance(r=radius, overlap=overlap)
    print(f"Each block is {distance} apart")

    origin_1 = train_positions[0]
    centroids = []

    # 找到第一个圆心
    temp_origin = {}
    for index, origin in enumerate(img_train_origins_resort):
        # 第一个块所在的圆心
        if np.linalg.norm(origin_1 - img_train_origins_resort[origin]) > radius:
            centroids.append(temp_origin)
            break
        temp_origin = origin

    #从当前圆心开始，每隔distance取一个点作为新的圆心
    temp_origin = {}
    judge = False
    for idx, origin in enumerate(img_train_origins_resort):
        if origin != centroids[-1] and judge == False:  # 还没有到达第一个圆心
            continue
        else:
            judge = True
        if np.linalg.norm(img_train_origins_resort[centroids[-1]] - img_train_origins_resort[origin]) > distance:
            centroids.append(temp_origin)
        temp_origin = origin

    split_train_results = {}
    for idx, centroid in enumerate(centroids):
        # 根据每个块的中心找到半径范围内的点
        block_train_element = get_each_block_element_train(img_train_origins_resort, centroid, radius)
        block_info = {
            'elements': block_train_element,
            "centroid": [centroid, img_train_origins_resort[centroid].tolist()]
        }
        split_train_results[f'block_{idx}'] = block_info
        # 提取出了当前block的所有train数据，现在要根据train数据的index确定val的index
        # val的index和所属block应该是同一相机情况下离他最近的那个
        # block_val_element=get_each_block_element_val(img_val_origins_resort,centroids,radius)
    # 对每一个val_origin找到其同一相机下最近的距离
    split_val_results = get_val_block_index(img_val_origins_resort, train_meta, val_meta, img_train_origins_resort,
                                            split_train_results)

    return split_train_results, split_val_results


def extract_origins(meta):
    origins = []
    cam_index = defaultdict(int)
    for img_name in meta:
        img_info = meta[img_name]
        origins.append(img_info['origin_pos'])
        cam_index[img_info['cam_idx']] += 1

    print(cam_index)
    origins = np.array(origins)
    return origins


def visualize_origin(train_origins, val_origins, block_info=None, radius=2, visual_Block=True):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    train_pcd = o3d.geometry.PointCloud()
    train_pcd.points = o3d.utility.Vector3dVector(train_origins)
    train_pcd.paint_uniform_color([0, 1, 0])
    val_pcd = o3d.geometry.PointCloud()
    val_pcd.points = o3d.utility.Vector3dVector(val_origins)
    val_pcd.paint_uniform_color([1, 0, 0])

    print(train_pcd, val_pcd)

    draw_list = [train_pcd, val_pcd, coord_frame]

    if visual_Block:
        if block_info != None:
            for block in block_info:
                print(f"{block} has {len(block_info[block]['elements'])} images!")
                block = block_info[block]
                pos = np.array(block['centroid'][1])
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
                sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)  # 网格
                sphere.translate(pos)  # 首先与第一维对齐
                sphere.paint_uniform_color((random.random(), random.random(), random.random()))
                draw_list.append(sphere)

    # o3d.visualization.draw_geometries([pcd])  # ,coord_frame])
    o3d.visualization.draw_geometries(draw_list)


if __name__ == "__main__":
    args = get_hparams()
    print(args)
    root_dir = "../data/WaymoDataset"
    save_dir = "../data/WaymoDataset/json"

    os.makedirs(save_dir, exist_ok=True)
    train_meta_path = os.path.join(save_dir, "split_block_train.json")
    val_meta_path = os.path.join(save_dir, "split_block_val.json")

    with open(os.path.join(root_dir, 'json/train.json'), 'r') as fp:
        train_meta = json.load(fp)

    with open(os.path.join(root_dir, 'json/val.json'), 'r') as fp:
        val_meta = json.load(fp)

    print(
        f"Before spliting, there are {len(train_meta)} images for train and {len(val_meta)} images for val!")

    # 输入train_origins,val_origins,然后返回split_block_train和split_block_val
    # 分别为同一个Block中的train和val数据
    split_train_results, split_val_results = split_dataset(train_meta, val_meta, radius=args['radius'],
                                                           overlap=args['overlap'])
    print("Complete the split work!")

    block_train_json = {}
    block_val_json = {}

    for block in split_train_results:
        block_train_json[block] = split_train_results[block]
        print(f"{block} has {len(split_train_results[block]['elements'])}")
        with open(train_meta_path, "w") as fp:
            json.dump(block_train_json, fp)
            fp.close()

    for block in split_val_results:
        block_val_json[block] = split_val_results[block]
        print(f"{block} has {len(split_val_results[block])}")
        with open(val_meta_path, "w") as fp:
            json.dump(block_val_json, fp)
            fp.close()

    print(f"The split results has been stored in the {train_meta_path} and {val_meta_path}")

    if args['visualization']:
        train_origins = extract_origins(train_meta)  # 只提取pos
        val_origins = extract_origins(val_meta)
        # 11269,3
        # block_json
        visualize_origin(train_origins, val_origins, block_info=block_train_json, radius=args['radius'],
                         visual_Block=args['visual_Block'])
