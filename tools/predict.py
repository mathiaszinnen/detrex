import os
import json

from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.structures.boxes import BoxMode
from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_categories(json_f):
    with open(json_f) as f:
        coco = json.load(f)
    return coco['categories']

def to_coco(dt2_output, categories):
    imgs = []
    annotations = []

    for img, preds in dt2_output:
        imgs.append({
            'id': img['id'],
            'file_name': img['file_name'],
            'height': img['height'],
            'width': img['width']
        })
        boxes = preds.get('pred_boxes')
        scores = preds.get('scores')
        classes = preds.get('pred_classes')

        for box, score, cls in zip(boxes, scores, classes):
            ann_id = len(annotations)
            box = box.cpu().numpy()
            box = BoxMode.convert(np.array([box]), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            annotations.append(
                {
                    'id': ann_id,
                    'image_id': img['id'],
                    'category_id': int(cls.cpu().item()),
                    'bbox': list(map(int, box[0])),
                    'score': float(score.cpu().item()),
                    'area': int(box[0][2] * box[0][3])
                }
            )

    return {
        'images': imgs,
        'annotations': annotations,
        'categories': categories
    }


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    
    out_folder = 'coco-predictions/detrex'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_path = os.path.join(out_folder, 'odeuropa_40k_dummy.json')

    device = cfg.train.device
    model = instantiate(cfg.model)
    model.to(device)
    checkpointer = DetectionCheckpointer(model)
    print(f"Loading checkpoint: {cfg.train.final_checkpoint}")
    checkpointer.load(cfg.train.final_checkpoint)

    model.eval()

    dataloader = instantiate(cfg.dataloader.inference)

    output = []
    index = 0
    for img in tqdm(dataloader):
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            model_preds = model(img)
        output.append((img[0], model_preds[0]['instances']))
        index += 1
        if index % 5000 == 0:
            out = to_coco(output, categories=load_categories(cfg.train.valid_json))
            with open(out_path, 'w') as f:
                json.dump(out, f)        

    out = to_coco(output, categories=load_categories(cfg.train.valid_json))
    with open(out_path, 'w') as f:
        json.dump(out, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
