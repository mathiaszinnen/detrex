from detrex.config import get_config
from detectron2.config import LazyCall as L
from detectron2.modeling import ShapeSpec
from detectron2.layers import FrozenBatchNorm2d

from .dino_r50_4scale_12ep import (
    train,
    optimizer,
)
from .models.dino_r50 import model

from detrex.modeling.backbone import TimmBackbone

# modify backbone configs
model.backbone = L(TimmBackbone)(
    model_name="resnet50",  # name in timm
    features_only=True,
    pretrained=False,
    in_channels=3,
    out_indices=(1, 2, 3),
    norm_layer=FrozenBatchNorm2d,
)

# modify neck configs
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=256),
    "p2": ShapeSpec(channels=512),
    "p3": ShapeSpec(channels=1024),
}
model.neck.in_features = ["p1", "p2", "p3"]

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

EPOCH_ITERS = 4264

dataloader = get_config("common/data/odor.py").dataloader
model.num_classes = 139

train.max_iter = 50 * EPOCH_ITERS
train.output_dir = "/home/woody/iwi5/iwi5064h/backbone_ablation/rn50_in1k_timm"

train.eval_period = EPOCH_ITERS
train.checkpointer.period = 10 * EPOCH_ITERS

train.final_checkpoint = "/home/woody/iwi5/iwi5064h/backbone_ablation/rn50_in1k_timm/final"

#model.backbone.drop_path_rate = 0.4

# modify training configs
train.init_checkpoint = "/home/vault/iwi5/iwi5064h/timm_weights/resnet50_a1_0-14fe96d1.pth"
