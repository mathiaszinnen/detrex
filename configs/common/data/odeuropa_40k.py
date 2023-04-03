import itertools, os
import json
import pandas as pd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog,
)
from detectron2.structures import BoxMode

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()




dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="odeuropa_40k_test", filter_empty=False),
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

# dataloader.evaluator = L(COCOEvaluator)(
#     dataset_name="${..test.dataset.names}",
# )


def get_odor_dict_test():
    return get_odor_dict('test')


def get_odor_meta(split):
    if split == 'test':
        pth = 'data/ODOR-v3/clip-style/mmodor_complete_title_keywords_s3.csv'
        json_pth = 'data/ODOR-v3/coco-style/annotations/instances_val2017.json'
        img_pth = '/media/prathmeshmadhu/myhdd/odeuropa/annotations-nightly/mmodor_imgs'
    else:
        raise Exception

    with open(json_pth) as f:
        coco = json.load(f)
    class_names = [cat['name'] for cat in coco['categories']]

    return {
        "csv_path": pth,
        "json_file": json_pth,
        "image_root": img_pth,
        "class_names": class_names
    }


def get_odor_dict(split):
    meta = get_odor_meta(split)

    pth = meta['csv_path']
    img_pth = meta['image_root']

    coco_anns = pd.read_csv(pth)
    records = []

    for i, row in coco_anns.iterrows():
        img = {}
        img['file_name'] = f'{img_pth}/{row["File Name"]}'
        img['id'] = row["image_id"]
        # records.append(img)
        if os.path.exists(img['file_name']):
            records.append(img)
        else:
            continue
    print(len(records))
    return records

# import pdb; pdb.set_trace()
DatasetCatalog.register('odeuropa_40k_test', get_odor_dict_test)

meta_test = get_odor_meta('test')

for split in ['test']:
    ds_name = f'odeuropa_40k_{split}'
    meta = get_odor_meta(split)

    MetadataCatalog.get(ds_name).set(
        thing_classes=meta['class_names'], json_file=meta['json_file'], image_root=meta['image_root'], evaluator_type='coco')
