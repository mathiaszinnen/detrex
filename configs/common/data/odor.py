import itertools
import json

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

register_coco_instances("odor_train", {}, 'data/ODOR-v3/coco-style/annotations/instances_train2017.json', 'data/ODOR-v3/coco-style/train2017')
register_coco_instances("odor_test", {}, 'data/ODOR-v3/coco-style/annotations/instances_val2017.json', 'data/ODOR-v3/coco-style/val2017')

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="odor_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="odor_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


def get_odor_dict(split):
    if split == 'train':
        pth = 'data/ODOR-v3/coco-style/annotations/instances_train2017.json'
    elif split == 'test':
        pth = 'data/ODOR-v3/coco-style/annotations/instances_val2017.json'
    else:
        raise Exception

    with open(pth) as f:
        coco_anns = json.load(f)

    records = []

    imid_to_anns = {}
    for ann in coco_anns['annotations']:
        if ann['image_id'] not in imid_to_anns.keys():
            imid_to_anns[ann['image_id']] = [ann]
        else:
            im_anns = imid_to_anns[ann['image_id']]
            im_anns.append(ann)
            imid_to_anns[ann['image_id']] = im_anns

    for img in coco_anns['images']:
        im_anns = imid_to_anns[img['id']]
        for ann in im_anns:
            ann['bbox_mode'] = BoxMode.XYWH_ABS
            ann['segmentation'] = []
        img['annotations'] = im_anns
        records.append(img)

    return records

for split in ['train', 'test']:
    DatasetCatalog.register(f'odor_{split}', get_odor_dict(split))
    MetadataCatalog.get(f'odor_{split}').set(thing_classes=["abc,def"])
