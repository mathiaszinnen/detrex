from detrex.config import get_config

from .dino_focalnet_large_lrf_384_fl4_5scale_12ep import (
    train,
    model,
)

EPOCH_ITERS = 4264

dataloader = get_config("common/data/odeuropa_40k.py").dataloader
model.num_classes = 139

# using 36ep scheduler
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep


# modify training config
train.valid_json = "data/ODOR-v3/coco-style/annotations/instances_val2017.json"
train.final_checkpoint = "trained-models/dino-focal/focaldino_ep18.pth"

