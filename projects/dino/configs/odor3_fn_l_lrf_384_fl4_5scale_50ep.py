from detrex.config import get_config

from .dino_focalnet_large_lrf_384_fl4_5scale_12ep import (
    train,
    optimizer,
    model,
)

EPOCH_ITERS = 4264

dataloader = get_config("common/data/odor.py").dataloader
model.num_classes = 139

# using 36ep scheduler
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

# modify training config
train.max_iter = 213200
#train.max_epoch = 50
train.init_checkpoint = "/home/woody/iwi5/iwi5064h/focalnet_large_lrf_384_fl4.pth"
train.output_dir = "/home/woody/iwi5/iwi5064h/dino-focalnet"

train.eval_period = EPOCH_ITERS
train.checkpointer.period = EPOCH_ITERS
train.final_checkpoint = "trained-models/dino-focal/focaldino_ep18.pth"

# using larger drop-path rate for longer training times
model.backbone.drop_path_rate = 0.4
