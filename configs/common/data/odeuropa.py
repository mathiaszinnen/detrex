import itertools
import json
import pandas as pd

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

dataloader.inference = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="odeuropa_europeana_v1", filter_empty=False),
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


def get_odor_meta(split):
    if split == 'inference':
        pth = 'data/ODOR-v3/clip-style/mmodor_complete_title_keywords_s1.csv'
        img_pth = '../annotations-nightly/mmodor_imgs'
    else:
        raise Exception

    return {
        "csv_file": pth,
        "image_root": img_pth,
    }


def get_odor_dict_inference():
    return get_odor_dict('inference')


def get_odor_dict(split):
    meta = get_odor_meta(split)

    pth = meta['csv_file']
    img_pth = meta['image_root']

    coco_anns = pd.read_csv(pth)

    records = []

    imid_to_anns = {}
    
    for i, row in coco_anns.iterrows():
        img = {}
        img['file_name'] = f'{img_pth}/{row["File Name"]}'
        img['id'] = row['image_id']
        records.append(img)

    return records


DatasetCatalog.register('odeuropa_europeana_v1', get_odor_dict_inference)

meta_inference = get_odor_meta('inference')


ds_name = f'odeuropa_europeana_v1'
meta = get_odor_meta("inference")

MetadataCatalog.get(ds_name).set(
    json_file=meta['csv_file'], image_root=meta['image_root'], evaluator_type='coco')


# MetadataCatalog.get(ds_name).set(
#     thing_classes=meta['class_names'], json_file=meta['json_file'], image_root=meta['image_root'], evaluator_tyoe='coco')
