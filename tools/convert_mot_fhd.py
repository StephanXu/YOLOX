from typing import Tuple

import os
import os.path as osp
import configparser
import pandas as pd
import tqdm
import shutil
import json
import cv2

def region_origin_points(image_size: Tuple[int, int], input_size: Tuple[int, int], row: int, col: int, img=None):
    height, width = image_size
    input_height, input_width = input_size
    res = []
    for x in range(0, width - input_width + 1, (width - input_width) // (col - 1)):
        for y in range(0, height - input_height + 1, (height - input_height) // (row - 1)):
            res.append((x, y, None if not img  else img[y:y + input_height, x:x + input_width]))
    return res

def _get_tracklet_categories(tracklet_name: str):
    if tracklet_name.find('player') != -1:
        return 1
    elif tracklet_name.find('goalkeeper') != -1:
        return 2
    elif tracklet_name.find('referee') != -1:
        return 3
    elif tracklet_name.find('other') != -1:
        return 4
    elif tracklet_name.find('ball') != -1:
        return 5
    raise Exception(f"Unexpected tracklet name: {tracklet_name}")


categories = [
    dict(supercategory='person', id=1, name='player'),
    dict(supercategory='person', id=2, name='goalkeeper'),
    dict(supercategory='person', id=3, name='referee'),
    dict(supercategory="person", id=4, name="cast"),
    dict(supercategory='others', id=5, name='ball'),
]

def convert_dataset(base_dir, output_dir, dataset_name, stride = 1):
    images = []
    annotations = []

    output_images_dir = osp.join(output_dir, dataset_name)
    id_counter = 0
    annotation_id = 0
    os.makedirs(output_images_dir)
    annotations_dir = osp.join(output_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    for dir_name in tqdm.tqdm(os.listdir(base_dir)):
        sample_base = osp.join(base_dir, dir_name)

        # ====================
        # parse meta data
        # ====================
        game_info_path = os.path.join(sample_base, 'gameinfo.ini')
        seq_info_path = os.path.join(sample_base, 'seqinfo.ini')

        seq_info = configparser.ConfigParser()
        seq_info.read(seq_info_path)
        frame_rate = seq_info.getint("Sequence", 'frameRate')
        seq_name = seq_info.get('Sequence', 'name')
        im_ext = seq_info.get('Sequence', 'imExt')
        im_dir = seq_info.get('Sequence', 'imDir')
        im_width = seq_info.getint('Sequence', 'imWidth')
        im_height = seq_info.getint('Sequence', 'imHeight')

        gt_path = os.path.join(sample_base, 'gt', 'gt.txt')
        frame_dir = os.path.join(sample_base, im_dir)
        gt = pd.read_csv(gt_path, header=0, names=[
                         'frame', 'track', 'x', 'y', 'w', 'h', 'c', 'r1', 'r2', 'r3'])

        game_info = configparser.ConfigParser()
        game_info.read(game_info_path)
        tracklet_count = game_info.getint('Sequence', 'num_tracklets')
        tracklets = dict([
            (i, _get_tracklet_categories(game_info.get('Sequence', f'trackletID_{i}'))) for i in range(1, tracklet_count + 1)
        ])

        # ====================
        # convert annoations
        # ====================
        input_size = (640, 640)
        input_height, input_width = input_size
        for img_filename in tqdm.tqdm(os.listdir(frame_dir)):
            origin_img = cv2.imread(osp.join(frame_dir, img_filename))
            for base_x, base_y, crop in region_origin_points((im_height, im_width), input_size, 4, 2, origin_img):
                id_counter += 1
                if id_counter % stride != 0:
                    continue

                crop_r, crop_b = base_x + input_width, base_y + input_height
                output_image_filename = f'{str(id_counter).zfill(12)}.jpg'
                cv2.imwrite(osp.join(output_images_dir, output_image_filename), crop)

                images.append(dict(filename=output_image_filename,
                                   height=input_height,
                                   width=input_width,
                                   id=id_counter))
                
                # generate instances
                for _, obj in gt[(gt.frame == int(img_filename.replace(im_ext, '')))].iterrows():
                    l, t, w, h = map(int, (obj['x'], obj['y'], obj['w'], obj['h']))
                    r, b = l + w, t + h
                    if l + w < base_x or l > crop_r:
                        continue
                    if t + h < base_y or t > crop_b:
                        continue
                    l = max(base_x, l)
                    t = max(base_y, t)
                    w = min(crop_r, r) - l
                    h = min(crop_b, b) - t
                    l, t = l - base_x, t - base_y
                    r, b = l + w, t + h

                    annotation_id += 1
                    annotations.append(dict(category_id=tracklets[obj['track']],
                                            image_id=id_counter,
                                            bbox=list(map(int, (l, t, w, h))),
                                            iscrowd=0,
                                            id=annotation_id,
                                            segmentation=[l, t, l, b, r, b, r, t],
                                            area=w * h))
    label = {
        'images': images,
        'categories': categories,
        'annotations': annotations
    }
    with open(osp.join(annotations_dir, f'instances_{dataset_name}.json'), 'w') as f:
        json.dump(label, f)
    pass

def main():
    convert_dataset(base_dir='D:\\stephanxu\\soccer-net-tracking\\tracking\\train',
                    output_dir='D:\\stephanxu\\workspace\\YOLOX\\datasets\\soccer',
                    dataset_name='train',
                    stride=5)
    convert_dataset(base_dir='D:\\stephanxu\\soccer-net-tracking\\tracking\\test',
                    output_dir='D:\\stephanxu\\workspace\\YOLOX\\datasets\\soccer',
                    dataset_name='test',
                    stride=5)

if __name__ == '__main__':
    main()
