from __future__ import absolute_import, division, print_function

import ast
import glob
import os
import sys
import warnings
import numpy as np

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
grandparent_dir = os.path.join(os.path.dirname(parent_path), 'PaddleClas-2.5.2')
sys.path.insert(0, grandparent_dir)

import paddle
from PIL import Image, ImageOps, ImageDraw
from ppcls.engine.engine import Engine
from ppcls.utils import config
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer, Trainer_ARSL
from ppdet.metrics import get_infer_results
from ppdet.slim import build_slim_model
from ppdet.utils.check import check_config, check_gpu, check_version
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.utils.logger import setup_logger

warnings.filterwarnings('ignore')
logger = setup_logger('train')

cls_categories = ['helmet', 'smoke', 'playphone']
multi_threshold = 0.4


"""
python3 tools/detect.py \
-c /work/PaddleDetection-2.8.1/data/helmet/det.yml \
-o weights=/work/Demo/det.pdparams \
--cls_config /work/Demo/cls.yml \
--cls_override Global.pretrained_model=/work/Demo/cls \
--infer_dir /work/Demo/imgs \
--output_dir /work/Demo/output
"""


def parse_args():
    parser = ArgsParser()
    parser.add_argument('--infer_dir', type=str, default=None, help='Directory for images to perform inference on.')
    parser.add_argument('--infer_img', type=str, default=None, help='Image path, has higher priority over --infer_dir')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for storing visualization files.')
    parser.add_argument('--draw_threshold', type=float, default=0.5, help='Threshold to reserve visualization.')
    parser.add_argument('--cls_config', type=str, default=None, help='')
    parser.add_argument('--cls_override', type=str, default=None, help='')
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, '--infer_img or --infer_dir should be set'
    assert infer_img is None or os.path.isfile(infer_img), f'{infer_img} is not a file'
    assert infer_dir is None or os.path.isdir(infer_dir), f'{infer_dir} is not a directory'

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), f'infer_dir {infer_dir} is not a directory'
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob(f'{infer_dir}/*.{ext}'))
    images = list(images)

    assert len(images) > 0, f'no image found in {infer_dir}'
    logger.info(f'Found {len(images)} inference images in total.')

    return images


def _get_save_image_name(output_dir, image_path):
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, f'{name}') + ext


def infer(self, image):
    assert self.mode == 'infer' and self.eval_mode == 'classification'
    assert isinstance(image, Image.Image), 'must be PIL.Image'

    import io

    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    x = buf.getvalue()
    for process in self.preprocess_func:
        x = process(x)
    if isinstance(x, paddle.Tensor):
        batch_tensor = x.unsqueeze(0)
    else:
        batch_tensor = paddle.to_tensor(x).unsqueeze(0)

    self.model.eval()
    if self.amp and self.amp_eval:
        with paddle.amp.auto_cast(custom_black_list={'flatten_contiguous_range', 'greater_than'}, level=self.amp_level):
            out = self.model(batch_tensor)
    else:
        out = self.model(batch_tensor)
    if isinstance(out, list):
        out = out[0]
    if isinstance(out, dict):
        if 'Student' in out:
            out = out['Student']
        elif 'logits' in out:
            out = out['logits']
        elif 'output' in out:
            out = out['output']
    dummy_path_list = ['infer_image']
    return self.postprocess_func(out, dummy_path_list)


def nms_numpy(boxes, scores, iou_thresh=0.5):
    """
    boxes: (N,4) x1,y1,x2,y2
    scores: (N,)
    return: indices of kept boxes (relative to input order), sorted by decreasing score
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


def predict_asst(cls_engine, image, bbox_res):
    if bbox_res is None or len(bbox_res) == 0:
        return
    boxes = []
    scores = []
    src_idxs = []
    for idx, b in enumerate(bbox_res):
        if b.get('category_id', -1) != 0:
            continue
        score = float(b.get('score', 0.0))
        if score <= 0.5:
            continue
        bbox = b.get('bbox', None)
        if bbox is None:
            continue
        x = float(bbox[0])
        y = float(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        src_idxs.append(idx)
    if len(boxes) == 0:
        return
    boxes_np = np.array(boxes, dtype=float)
    scores_np = np.array(scores, dtype=float)
    keep_inds = nms_numpy(boxes_np, scores_np, iou_thresh=0.5)
    draw = ImageDraw.Draw(image)
    for k in keep_inds:
        x1, y1, x2, y2 = boxes_np[k]
        x1_i = max(0, int(round(x1)))
        y1_i = max(0, int(round(y1)))
        x2_i = min(image.width, int(round(x2)))
        y2_i = min(image.height, int(round(y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue
        crop = image.crop((x1_i, y1_i, x2_i, y2_i))
        clsres_crop = cls_engine.infer_image(crop)
        per_class_scores = [0.0] * len(cls_categories)
        if isinstance(clsres_crop, (list, tuple)) and len(clsres_crop) > 0:
            first = clsres_crop[0]
            ids = first.get('class_ids', [])
            scs = first.get('scores', [])
            for cid, sc in zip(ids, scs):
                try:
                    per_class_scores[int(cid)] = float(sc)
                except Exception:
                    pass
        label_parts = []
        for idx, cname in enumerate(cls_categories):
            if per_class_scores[idx] > multi_threshold:
                label_parts.append(cname)
            else:
                label_parts.append('not_' + cname)

        if x2_i - x1_i > (y2_i - y1_i) * 2:
            label_parts.append('falling')
        else:
            label_parts.append('not_falling')

        # label_summary = ','.join(label_parts)
        # det_score = float(scores_np[k])
        # print(
        #     f'det: [{x1_i}, {y1_i}, {x2_i}, {y2_i}], det_score={det_score:.4f}; cls={label_summary}, cls_score={per_class_scores}'
        # )

        box_color = (255, 0, 0)
        text_color = (255, 255, 255)
        pad = 4

        draw.rectangle([x1_i, y1_i, x2_i, y2_i], outline=box_color, width=3)

        text_lines = label_parts
        try:
            from PIL import ImageFont

            font = ImageFont.load_default()
        except Exception:
            font = None
        line_sizes = []
        for line in text_lines:
            if font is not None:
                try:
                    size = font.getsize(line)
                except Exception:
                    try:
                        bbox = draw.textbbox((0, 0), line, font=font)
                        size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    except Exception:
                        size = (len(line) * 6, 11)
            else:
                try:
                    bbox = draw.textbbox((0, 0), line)
                    size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                except Exception:
                    size = (len(line) * 6, 11)
            line_sizes.append(size)

        text_w = max(w for w, h in line_sizes)
        text_h = sum(h for w, h in line_sizes) + (len(text_lines) - 1) * 2

        bg_x1 = x1_i + pad
        bg_y1 = y1_i + pad
        bg_x2 = min(x2_i - pad, bg_x1 + text_w + 2 * pad)
        bg_y2 = min(y2_i - pad, bg_y1 + text_h + 2 * pad)

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=box_color)
        cur_y = bg_y1 + pad
        for line, (lw, lh) in zip(label_parts, line_sizes):
            if font is not None:
                draw.text((bg_x1 + pad, cur_y), line, fill=text_color, font=font)
            else:
                draw.text((bg_x1 + pad, cur_y), line, fill=text_color)
            cur_y += lh + 2


def run(FLAGS, cfg):
    # cls model
    cls_engine = Engine(config.get_config(FLAGS.cls_config, [FLAGS.cls_override]), mode='infer')
    # det model
    ssod_method = cfg.get('ssod_method', None)
    if ssod_method == 'ARSL':
        det_engine = Trainer_ARSL(cfg, mode='test')
        det_engine.load_weights(cfg.weights, ARSL_eval=True)
    else:
        det_engine = Trainer(cfg, mode='test')
        det_engine.load_weights(cfg.weights)
    # get inference images
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset = create('TestDataset')()
    dataset.set_images(images, do_eval=False)
    imid2path = dataset.get_imid2path()
    clsid2catid = {i: i for i in range(cfg.get('num_classes', 0))}
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    # inference
    results = det_engine.predict(images, visualize=False)
    for outs in results:
        batch_res = get_infer_results(outs, clsid2catid)
        bbox_num = outs['bbox_num']
        start = 0
        for i, im_id in enumerate(outs['im_id']):
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')
            image = ImageOps.exif_transpose(image)
            end = start + bbox_num[i]
            bbox_res = batch_res['bbox'][start:end] if 'bbox' in batch_res else None
            predict_asst(cls_engine, image, bbox_res)
            # save image with detection
            save_name = _get_save_image_name(FLAGS.output_dir, image_path)
            logger.info(f'Detection bbox results save in {save_name}')
            image.save(save_name, quality=95)
            start = end


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)
    paddle.set_device('gpu')
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
