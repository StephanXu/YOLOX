import os
import os.path as osp
import configparser
import pandas as pd
import tqdm
import shutil
import json


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
        for img_filename in tqdm.tqdm(os.listdir(frame_dir)):
            id_counter += 1
            if id_counter % stride != 0:
                continue

            output_image_filename = f'{str(id_counter).zfill(12)}.jpg'

            # copy image
            shutil.copy(osp.join(frame_dir, img_filename), osp.join(
                output_images_dir, output_image_filename))
            images.append(dict(filename=output_image_filename,
                          height=im_height, width=im_width, id=id_counter))

            # generate instances
            for _, obj in gt[gt['frame'] == int(img_filename.replace(im_ext, ''))].iterrows():
                annotation_id += 1
                l, t, w, h = map(int, (obj['x'], obj['y'], obj['w'], obj['h']))
                annotations.append(dict(category_id=tracklets[obj['track']],
                                        image_id=id_counter,
                                        bbox=list(
                                            map(int, (obj['x'], obj['y'], obj['w'], obj['h']))),
                                        iscrowd=0,
                                        id=annotation_id,
                                        segmentation=[l, t, l, t + h, l + w, t + h, l + w, t],
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
