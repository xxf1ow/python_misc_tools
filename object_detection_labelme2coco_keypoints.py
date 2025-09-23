import json
import os
import uuid

import numpy as np
import PIL.Image
from tqdm import tqdm

from module import checkCOCO, find_dir, find_img, parse_labelimg, parse_labelme

##################################################################
#
#   此文件用于关键点检测数据集转换格式, 从 labelme 多边形标注转为 COCO 格式, 用于关键点检测训练
#
##################################################################

detection_class = 'Scale'
keypoints_class = ['beg_tl', 'beg_br', 'end_tl', 'end_br', 'point']
shape_check_keys = ['beg_tl', 'beg_br', 'end_br', 'end_tl']  # 用于按顺序检查标注的正确性
skeleton = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]]


# 判断点 point 是否在矩形 rect 内部, 宽松版本. rect: [xmin, ymin, xmax, ymax]
def rectangle_include_point_wide(r, p, w):
    return p[0] >= r[0] - w and p[0] <= r[2] + w and p[1] >= r[1] - w and p[1] <= r[3] + w


def sort_rotation_points(points):
    """scale --> beg_tl, beg_br, end_br, end_tl"""

    # 1. 输入校验与格式转换
    if len(points) != 8:
        raise ValueError('输入数组必须正好包含8个元素(4个点的坐标).')

    # 将扁平数组转换为 4x2 的点坐标数组
    pts = np.array(points, dtype=float).reshape(4, 2)

    # 找到矩形的两个边向量
    p0 = pts[0]
    dists_sq = np.sum((pts[1:] - p0) ** 2, axis=1)
    adjacent_indices = np.argsort(dists_sq)[:2] + 1
    v1 = pts[adjacent_indices[0]] - p0
    v2 = pts[adjacent_indices[1]] - p0

    # 确定长边向量 (主轴)
    v_long = v1 if np.sum(v1**2) > np.sum(v2**2) else v2

    # 标准化坐标轴方向:确保主轴指向右半平面或上半平面
    if v_long[0] < 0 or v_long[0] == 0 and v_long[1] < 0:
        v_long = -v_long

    # 按长边投影将点分为两组 (每组是短边上的两个点)
    sorted_by_proj = sorted(pts, key=lambda p: np.dot(p, v_long))
    group1 = np.array(sorted_by_proj[:2])
    group2 = np.array(sorted_by_proj[2:])

    # 根据全局坐标系定义判断哪个是 beg/end 组
    center1 = np.mean(group1, axis=0)
    center2 = np.mean(group2, axis=0)
    if (center1[0] - center1[1]) > (center2[0] - center2[1]):
        end_group, beg_group = group1, group2
    else:
        end_group, beg_group = group2, group1

    # 组内排序：tl点x+y更小，br点x+y更大
    def sort_group(group):
        sum_coords = np.sum(group, axis=1)
        if sum_coords[0] < sum_coords[1]:
            return group[0], group[1]  # tl, br
        else:
            return group[1], group[0]  # tl, br

    end_tl, end_br = sort_group(end_group)
    beg_tl, beg_br = sort_group(beg_group)

    # 返回与输入相同的格式
    return np.array([beg_tl, beg_br, end_br, end_tl]).reshape(-1).tolist()


def rotation_to_skeleton(original, width, height):
    if len(original) != 8:
        raise ValueError('输入列表必须恰好包含 8 个坐标值(4个点)')
    processed = original.copy()

    # 画面的四条边
    rect_edges = [[0, 0, width, 0], [width, 0, width, height], [width, height, 0, height], [0, height, 0, 0]]

    def _approximate_length(line):
        dx = line[0] - line[2]
        dy = line[1] - line[3]
        return dx * dx + dy * dy

    def _intersection(l1, l2, eps=1e-6):
        l1dx = l1[2] - l1[0]
        l1dy = l1[3] - l1[1]
        l2dx = l2[2] - l2[0]
        l2dy = l2[3] - l2[1]
        cross = l1dx * l2dy - l2dx * l1dy
        if abs(cross) < eps:
            return False, -1, -1
        t = (l2dy * (l2[0] - l1[0]) - l2dx * (l2[1] - l1[1])) / cross
        u = (l1dy * (l2[0] - l1[0]) - l1dx * (l2[1] - l1[1])) / cross
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True, l1[0] + t * l1dx, l1[1] + t * l1dy
        return False, -1, -1

    for idx in range(4):
        ox, oy = original[idx * 2], original[idx * 2 + 1]
        if 0.0 <= ox <= width and 0.0 <= oy <= height:
            continue
        prev = [ox, oy, original[(idx - 1) % 4 * 2], original[(idx - 1) % 4 * 2 + 1]]
        next = [ox, oy, original[(idx + 1) % 4 * 2], original[(idx + 1) % 4 * 2 + 1]]
        line = prev if _approximate_length(prev) >= _approximate_length(next) else next

        inters = []
        for edge in rect_edges:
            intersects, px, py = _intersection(line, edge)
            if intersects:
                inters.append((px, py))

        if len(inters) == 0:
            processed[idx * 2] = min(max(ox, 0.0), width)
            processed[idx * 2 + 1] = min(max(oy, 0.0), height)
        else:
            inters_sorted = sorted(inters, key=lambda q: _approximate_length([ox, oy, q[0], q[1]]))
            processed[idx * 2] = inters_sorted[0][0]
            processed[idx * 2 + 1] = inters_sorted[0][1]

    return processed


# 单个图片
def generate(img_path, det_path, seg_path):
    # check image
    assert os.path.isfile(img_path), f'图片文件不存在: {img_path}'
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelimg anns file
    bbox = parse_labelimg(det_path, width, height)
    bbox = {instance: box for instance, box in bbox.items() if instance[0] == detection_class}
    # parse labelme anns file
    _, shapes = parse_labelme(seg_path, width, height, ['point', 'rotation'])

    # convert 'scale' to [beg_tl, beg_br, end_br, end_tl]
    shapes_add = {}
    shapes_remove = []
    for instance, shape in shapes.items():
        if instance[0] == 'scale':  # scale 是采用旋转框方法的四点标注
            shapes_remove.append(instance)
            pts = rotation_to_skeleton(sort_rotation_points(shape[0]), width, height)
            shapes_add[('beg_tl', uuid.uuid1())] = [[pts[0], pts[1]]]
            shapes_add[('beg_br', uuid.uuid1())] = [[pts[2], pts[3]]]
            shapes_add[('end_br', uuid.uuid1())] = [[pts[4], pts[5]]]
            shapes_add[('end_tl', uuid.uuid1())] = [[pts[6], pts[7]]]
        elif instance[0] not in keypoints_class:
            shapes_remove.append(instance)
            print(f'\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{instance[0]}, {img_path}\n')
    for key in shapes_remove:
        del shapes[key]
    shapes.update(shapes_add)

    # generate anns
    imgs_dict = dict(id=0, file_name=img_path, width=width, height=height)
    anns_dict = []
    for _, box in bbox.items():
        # 找到所有在框内的点 (同类别会覆盖)
        points = {
            instance[0]: shape[0]
            for instance, shape in shapes.items()
            if rectangle_include_point_wide(box, shape[0], 3)
        }
        if len(points) == 0:
            continue
        if len(points) < 5 or 'point' not in points:
            print('\n\n', box, '\n', points, '\n', shapes, '\n')
            print(f'\n\033[1;31m[Error] 框下无点: {img_path}\033[0m\n')
            exit()
            return {}, {}
        check0 = [coord for key in shape_check_keys if key in points for coord in points[key]]
        if len(check0) != 8:
            print('\n\n', box, '\n', points, '\n', shapes, '\n')
            print(f'\n\033[1;31m[Error] 标注点数错误: {img_path}\033[0m\n')
            exit()
            return {}, {}
        check1 = sort_rotation_points(check0)
        if check0 != check1:
            print('\n\n', points, '\n', check0, '\n', check1, '\n')
            print(f'\n\033[1;31m[Error] 标注顺序错误: {img_path}\033[0m\n')
            exit()
            return {}, {}
        # 将这些点按类别排序
        key_points = []
        for cls in keypoints_class:  # 2-可见不遮挡 1-遮挡 0-没有点
            key_points.extend([points[cls][0], points[cls][1], 2] if cls in points else [0, 0, 0])
        # 组成一个框的标签
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        annotation = dict(
            id=0,
            image_id=0,
            category_id=1,
            bbox=[box[0], box[1], box_w, box_h],
            segmentation=[],
            area=box_w * box_h,
            num_keypoints=len(points),
            keypoints=key_points,
            iscrowd=0,
        )
        anns_dict.append(annotation)
    return imgs_dict, anns_dict


# 创建 coco
def process(root_path, split, all_reserve=0, reserve_no_label=False):
    print('\n[info] start task...')
    # 定义类别
    cat = {
        'id': 0,
        'name': detection_class,
        'supercategory': detection_class,
        'keypoints': keypoints_class,
        'skeleton': skeleton,
    }
    data_train = dict(categories=[cat], images=[], annotations=[])  # 训练集
    data_test = dict(categories=[cat], images=[], annotations=[])  # 测试集
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
        img_list = find_img(imgs_dir_path)
        all_reserve_dir = len(img_list) < all_reserve
        not_ann_cnt = 0
        for num, file in enumerate(tqdm(img_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN')):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f'{dir}/imgs/{raw_name}{extension}'
            det_path = f'{dir}/anns/{raw_name}.xml'
            seg_path = f'{dir}/anns_seg/{raw_name}.json'
            # get dict
            imgs_dict, anns_dict = generate(img_path, det_path, seg_path)
            # check anns_dict size
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
    # export to file
    with open('./pose_train.json', 'w', encoding='utf-8') as f:
        json.dump(data_train, f, indent=4)
    checkCOCO('./pose_train.json')  # 检查COCO文件是否正确
    with open('./pose_test.json', 'w', encoding='utf-8') as f:
        json.dump(data_test, f, indent=4)
    checkCOCO('./pose_test.json')  # 检查COCO文件是否正确


if __name__ == '__main__':
    process(os.getcwd(), 10)
    print('\nAll process success\n')
