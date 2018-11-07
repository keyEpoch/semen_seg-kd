from __future__ import print_function
import os, sys
import argparse
import glob
import cv2


# info1={"dbInfo": {"frameID": -1, "vID": "ADE20k"}, "width": 683, "fpath_img": "ADEChallengeData2016/images/validation/ADE_val_00000001.jpg", "ade_scene": "abbey", "height": 512, "dbName": "ADE20k", "fpath_segm": "ADEChallengeData2016/annotations/validation/ADE_val_00000001.png"}
# info2={"dbInfo": {"frameID": -1, "vID": "ADE20k"}, "width": 683, "fpath_img": "ADEChallengeData2016/images/validation/ADE_val_00000001.jpg", "ade_scene": "abbey", "height": 512, "dbName": "ADE20k", "fpath_segm": "ADEChallengeData2016/annotations/validation/ADE_val_00000001.png"}
# print '\n'.join([str(info1),str(info2)])
def get_configs():
    return {
        'Dir_root': '',
        'Dataset_name': 'cityscapes',
        'Dataset_root': '',
        'img_dir_name': 'leftImg8bit',
        'gtcoarse_name': 'gtCoarse',
        'gtfine_name': 'gtFine',
        'img_root': '',
        'gtcoarse_root': '',
        'gtfine_root': '',
        'Split': ['train', 'val'],
        'img_no': '.png',
        'gt_no': '.png',
    }


def build_dict(img_path, gt_path):
    img_name = img_path.rstrip().split('/')[-1]
    gt_name = gt_path.rstrip().split('/')[-1]
    assert img_name == gt_name, 'name mismatch!'
    im = cv2.imread(img_path)
    height, width, _ = im.shape
    return {
        'dbInfo': {'frameID': -1, 'vID': 'Qijie'},
        'width': width,
        'height': height,
        'fpath_img': img_path,
        'fpath_segm': gt_path,
        'dbName': configs['Dataset_name'],
        'scene': img_path.split('/')[-2],
    }


if __name__ == '__main__':
    global configs
    configs = get_configs()
    configs['Dataset_root'] = os.path.join(configs['Dir_root'], configs['Dataset_name'])
    configs['img_root'] = os.path.join(configs['Dataset_root'], configs['img_dir_name'])
    configs['gtcoarse_root'] = os.path.join(configs['Dataset_root'], configs['gtcoarse_name'])
    configs['gtfine_root'] = os.path.join(configs['Dataset_root'], configs['gtfine_name'])

    print('============================Configs:==========================\n{}'.format(configs))
    print('=====>1, Split include: \n{}'.format(configs['Split']))
    Save_paths = dict()
    Fine = True
    print('=====>2, Select {} annotations'.format('Fine' if Fine == True else 'Coarse'))

    info_list = list()
    for split in configs['Split']:
        Save_paths[split] = os.path.join('data', '{}_{}.odgt'.format(configs['Dataset_name'], split))
        print('=====>3, Save path: \n{}'.format(Save_paths[split]))
        imgs = glob.glob(os.path.join(configs['img_root'], split, '*', '*{}'.format(configs['img_no'])))
        gt_root = configs['gtfine_root'] if Fine == True else configs['gtcoarse_root']
        gts = glob.glob(os.path.join(gt_root, split, '*', '*{}'.format(configs['gt_no'])))

        assert len(imgs) == len(gts), 'length mismatch! {} vs {}'.format(len(imgs), len(gts))
        print('=====>4, Collect {} imgs and gts in total.'.format(len(imgs)))

        imgs.sort()
        gts.sort()

        for i in range(len(imgs)):
            if i % 1000 == 0:
                print('### {} transfered.'.format(i))
            info_dict = build_dict(imgs[i], gts[i])
            info_list.append(info_dict)
        print('=====>5, {} has transfered done'.format(split))
        fw = open(Save_paths[split], 'w')
        fw.write('\n'.join([str(_).replace('\'', '\"') for _ in info_list]))
