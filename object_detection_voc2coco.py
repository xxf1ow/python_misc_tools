import json
import os

import PIL.Image
from tqdm import tqdm

from module import checkCOCO, find_dir, find_img, parse_labelimg

##################################################################
#
#   此文件用于目标检测数据集转换格式, 从 VOC 格式转为 COCO 格式
#
##################################################################

# 生成的数据集允许的标签列表
categories = ['D000', 'D001', 'P000', 'P001']

# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = set()


def downsample_list(lst, n):
    if n <= 0 or len(lst) <= n:
        return lst
    step = len(lst) / n
    indices = [round(i * step) for i in range(n)]
    return [lst[index] for index in indices]


# 单个图片
def generate(img_path, det_path):
    # check image
    assert os.path.isfile(img_path), f'图片文件不存在: {img_path}'
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelimg anns file
    bbox_dict = parse_labelimg(det_path, width, height)
    # generate anns
    imgs_dict = dict(id=0, file_name=img_path, width=width, height=height)
    anns_dict = []
    for instance, box in bbox_dict.items():
        label = instance[0]
        if label not in categories:
            skip_categories.add(label)
            continue
        label_id = categories.index(label)
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        annotation = dict(
            id=0,
            image_id=0,
            category_id=label_id,
            bbox=[box[0], box[1], box_w, box_h],
            segmentation=[],
            area=box_w * box_h,
            iscrowd=0,
        )
        anns_dict.append(annotation)
    return imgs_dict, anns_dict


def process(root_path, split, max_images=0, all_reserve=0, reserve_no_label=True):
    print('\n[info] start task...')
    data_train = dict(categories=[], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[], images=[], annotations=[])  # 测试集
    # 初始索引ID
    train_img_id = 0
    train_bbox_id = 0
    test_img_id = 0
    test_bbox_id = 0
    # 遍历脚本所在目录下的子文件夹
    for dir in find_dir(root_path):
        imgs_dir_path = os.path.join(root_path, dir, 'imgs')
        if not os.path.isdir(imgs_dir_path):
            continue
        img_list = downsample_list(find_img(imgs_dir_path), max_images)
        all_reserve_dir = len(img_list) < all_reserve
        not_ann_cnt = 0
        for num, file in enumerate(tqdm(img_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN')):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f'{dir}/imgs/{raw_name}{extension}'
            det_path = f'{dir}/anns/{raw_name}.xml'
            # 解析获取图片和标签字典
            imgs_dict, anns_dict = generate(img_path, det_path)
            # 无标注文件计数
            anns_size = len(anns_dict)
            not_ann_cnt += 1 if anns_size == 0 else 0
            if reserve_no_label is False and anns_size == 0:
                continue
            # train dataset
            if all_reserve_dir or split <= 0 or num % split != 0:
                imgs_dict['id'] = train_img_id
                data_train['images'].append(imgs_dict.copy())
                for idx, ann in enumerate(anns_dict):
                    ann['image_id'] = train_img_id
                    ann['id'] = train_bbox_id + idx
                    data_train['annotations'].append(ann.copy())
                train_img_id += 1
                train_bbox_id += anns_size
            # test dataset
            if all_reserve_dir or split <= 0 or num % split == 0:
                imgs_dict['id'] = test_img_id
                data_test['images'].append(imgs_dict.copy())
                for idx, ann in enumerate(anns_dict):
                    ann['image_id'] = test_img_id
                    ann['id'] = test_bbox_id + idx
                    data_test['annotations'].append(ann.copy())
                test_img_id += 1
                test_bbox_id += anns_size
        if not_ann_cnt != 0:
            print(f'\033[1;31m[Error] {dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m')
    print(f'\n训练集图片总数: {train_img_id}, 标注总数: {train_bbox_id}\n')
    print(f'测试集图片总数: {test_img_id}, 标注总数: {test_bbox_id}\n')
    # 导出到文件
    for id, category in enumerate(categories):
        cat = {'id': id, 'name': category, 'supercategory': category}
        data_train['categories'].append(cat)  # 训练集
        data_test['categories'].append(cat)  # 测试集
    with open('./train.json', 'w', encoding='utf-8') as f:
        json.dump(data_train, f, indent=4)
    checkCOCO('./train.json')  # 检查COCO文件是否正确
    with open('./test.json', 'w', encoding='utf-8') as f:
        json.dump(data_test, f, indent=4)
    checkCOCO('./test.json')  # 检查COCO文件是否正确


if __name__ == '__main__':
    process(os.getcwd(), 10)
    # 打印数据集中出现的不被允许的标签
    if len(skip_categories) > 0:
        print(f'\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}')
    print('\nAll process success\n')
