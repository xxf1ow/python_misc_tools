import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import PIL.Image
import PIL.ImageEnhance
from tqdm import tqdm

from module import create_labelimg, find_dir, find_img, parse_labelimg


##################################################################
#
#   此文件用于目标检测数据集数据增广
#
##################################################################
def augment_no_operation(image, bbox_dict):
    """
    无操作
    """
    return image, bbox_dict


def augment_vertical_flip(image, bbox_dict):
    """
    对图像及其边界框执行垂直(上下)翻转.
    """
    flipped_image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    width, height = image.size
    new_bbox_dict = {}
    for instance, box in bbox_dict.items():
        xmin, ymin, xmax, ymax = box  # box 格式应为 [xmin, ymin, xmax, ymax]
        new_ymin = height - ymax
        new_ymax = height - ymin
        new_box = [xmin, new_ymin, xmax, new_ymax]
        new_bbox_dict[instance] = new_box
    return flipped_image, new_bbox_dict


def augment_horizontal_flip(image, bbox_dict):
    """
    对图像及其边界框执行水平(左右)翻转.
    """
    flipped_image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    width, height = image.size
    new_bbox_dict = {}
    for instance, box in bbox_dict.items():
        xmin, ymin, xmax, ymax = box
        new_xmin = width - xmax
        new_xmax = width - xmin
        new_box = [new_xmin, ymin, new_xmax, ymax]
        new_bbox_dict[instance] = new_box
    return flipped_image, new_bbox_dict


def augment_rotate180(image, bbox_dict):
    """
    对图像及其边界框执行 180 度旋转, 等效于同时进行水平和垂直翻转
    """
    rotated_image = image.transpose(PIL.Image.ROTATE_180)
    width, height = image.size
    new_bbox_dict = {}
    for instance, box in bbox_dict.items():
        xmin, ymin, xmax, ymax = box
        new_xmin = width - xmax
        new_ymin = height - ymax
        new_xmax = width - xmin
        new_ymax = height - ymin
        new_box = [new_xmin, new_ymin, new_xmax, new_ymax]
        new_bbox_dict[instance] = new_box
    return rotated_image, new_bbox_dict


def augment_add_noise(image, bbox_dict):
    """
    向图像添加高斯噪声,边界框不变.
    """
    img_array = np.array(image)  # 将图像转为numpy数组
    gauss = np.random.normal(0, 25, img_array.shape)  # (均值, 标准差) 生成高斯噪声
    noisy_img_array = np.clip(img_array + gauss, 0, 255)  # 添加噪声并确保像素值在 [0, 255] 范围内
    noisy_image = PIL.Image.fromarray(noisy_img_array.astype('uint8'))  # 将数组转回图像
    return noisy_image, bbox_dict  # 噪声不影响边界框坐标


def augment_brighter_increasing(image, bbox_dict):
    """
    增加图像亮度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Brightness(image)
    brighter_image = enhancer.enhance(1.3)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


def augment_brighter_decreasing(image, bbox_dict):
    """
    减少图像亮度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Brightness(image)
    brighter_image = enhancer.enhance(0.7)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


def augment_contrast_increasing(image, bbox_dict):
    """
    增加图像对比度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Contrast(image)
    brighter_image = enhancer.enhance(1.3)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


def augment_contrast_decreasing(image, bbox_dict):
    """
    减少图像对比度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Contrast(image)
    brighter_image = enhancer.enhance(0.7)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


def augment_saturation_increasing(image, bbox_dict):
    """
    增加图像饱和度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Color(image)
    brighter_image = enhancer.enhance(1.3)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


def augment_saturation_decreasing(image, bbox_dict):
    """
    减少图像饱和度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Color(image)
    brighter_image = enhancer.enhance(0.7)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


def augment_sharpness_increasing(image, bbox_dict):
    """
    增加图像锐度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Sharpness(image)
    brighter_image = enhancer.enhance(1.3)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


def augment_sharpness_decreasing(image, bbox_dict):
    """
    减少图像锐度,边界框不变.
    """
    enhancer = PIL.ImageEnhance.Sharpness(image)
    brighter_image = enhancer.enhance(0.7)  # 增强因子,>1.0 表示增强
    return brighter_image, bbox_dict  # 颜色变化不影响边界框坐标


# 定义要应用的增广方法字典
AUG_POOLS = {
    # 方向池
    'direction': {
        augment_no_operation: 40,
        augment_vertical_flip: 20,
        augment_horizontal_flip: 20,
        augment_rotate180: 20,
    },
    # 色彩增强池
    'enhance': {
        augment_no_operation: 40,
        augment_brighter_increasing: 20,
        augment_contrast_increasing: 20,
        augment_saturation_increasing: 20,
    },
    # 色彩减弱池
    'weaken': {
        augment_no_operation: 40,
        augment_brighter_decreasing: 20,
        augment_contrast_decreasing: 20,
        augment_saturation_decreasing: 20,
    },
    # 锐度变化池
    'sharpness': {augment_no_operation: 40, augment_sharpness_increasing: 30, augment_sharpness_decreasing: 30},
    # 噪音池
    'noise': {augment_no_operation: 50, augment_add_noise: 50},
}


PRECOMPUTED_POOLS = {category: (list(pool.keys()), list(pool.values())) for category, pool in AUG_POOLS.items()}


def generate(img, ann, num_versions):
    ret = []
    for _ in range(num_versions):
        pipeline = []
        for category, (population, weights) in PRECOMPUTED_POOLS.items():
            chosen_func = random.choices(population, weights=weights, k=1)[0]
            pipeline.append(chosen_func)
        aug_img = img.copy()
        aug_ann = ann.copy()
        for aug_func in pipeline:
            aug_img, aug_ann = aug_func(aug_img, aug_ann)
        ret.append((aug_img, aug_ann))
    return ret


def process_image_task(dir, file, new_imgs_path, new_anns_path, num_versions, offset, idx):
    # misc path
    raw_name, extension = os.path.splitext(file)
    img_path = f'{dir}/imgs/{raw_name}{extension}'
    ann_path = f'{dir}/anns/{raw_name}.xml'
    # load img and parse labelimg anns file
    assert os.path.isfile(img_path), f'图片文件不存在: {img_path}'
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    ann = parse_labelimg(ann_path, width, height)
    # 应用增广
    if offset > 0:
        for num, (aug_img, aug_ann) in enumerate(generate(img, ann, num_versions)):
            filename = f'{num * offset + idx:06d}'
            aug_img.save(f'{new_imgs_path}/{filename}{extension}')
            create_labelimg(f'{new_anns_path}/{filename}.xml', file, width, height, aug_ann)
    else:
        for num, (aug_img, aug_ann) in enumerate(generate(img, ann, num_versions)):
            aug_img.save(f'{new_imgs_path}/{raw_name}_{num}{extension}')
            create_labelimg(f'{new_anns_path}/{raw_name}_{num}.xml', file, width, height, aug_ann)


def process(root_path, is_delivery, num_versions, max_workers=None):
    tasks = []
    print('\n[info] prepare task list ...')
    # 遍历脚本所在目录下的子文件夹
    for dir in find_dir(root_path):
        imgs_path = os.path.join(root_path, dir, 'imgs')
        anns_path = os.path.join(root_path, dir, 'anns')
        if not os.path.isdir(imgs_path) or not os.path.isdir(anns_path):
            continue
        new_imgs_path = os.path.join(root_path, 'generate', dir, 'imgs')
        new_anns_path = os.path.join(root_path, 'generate', dir, 'anns')
        os.makedirs(new_imgs_path, exist_ok=True)
        os.makedirs(new_anns_path, exist_ok=True)
        img_list = find_img(imgs_path)
        offset = len(img_list)
        # 将任务所需的所有参数打包成一个元组，添加到任务列表
        if is_delivery:
            for idx, file in enumerate(tqdm(img_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN')):
                tasks.append((dir, file, new_imgs_path, new_anns_path, num_versions, offset, idx))
        else:
            for file in tqdm(img_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN'):
                tasks.append((dir, file, new_imgs_path, new_anns_path, num_versions, 0, 0))

    print('\n[info] start task ...')
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image_task, *args) for args in tasks]
        with tqdm(total=len(tasks), leave=True, ncols=100, colour='CYAN') as pbar:
            for future in as_completed(futures):
                pbar.update(1)


if __name__ == '__main__':
    process(os.getcwd(), True, 8)
    print('\nAll process success\n')
