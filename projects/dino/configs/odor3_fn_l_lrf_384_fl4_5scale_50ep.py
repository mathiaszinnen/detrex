from detrex.config import get_config

from .dino_focalnet_large_lrf_384_fl4_5scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

model.num_classes=139

# using 36ep scheduler
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

# modify training config
train.max_iter = 270000
#train.max_epoch = 50
train.init_checkpoint = "/net/cluster/zinnen/models/focalnet_large_lrf_384_fl4.pth"
train.output_dir = "./net/cluster/zinnen/detrex-output/odor3-tests"

# using larger drop-path rate for longer training times
model.backbone.drop_path_rate = 0.4
