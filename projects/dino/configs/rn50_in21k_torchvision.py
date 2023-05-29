from detrex.config import get_config
from detectron2.config import LazyCall as L
from detectron2.modeling import ShapeSpec
from detectron2.layers import FrozenBatchNorm2d

from .dino_r50_4scale_12ep import (
    train,
    optimizer,
)
from .models.dino_r50 import model

from detrex.modeling.backbone import TorchvisionBackbone

# modify backbone configs
model.backbone = L(TorchvisionBackbone)(
    model_name="resnet50",  # name in timm
    # features_only=True,
    pretrained=False,
    return_nodes = {
        "layer2": "res3",
        "layer3": "res4",
        "layer4": "res5",
    }
)

# modify neck configs
model.neck.input_shapes = {
    "res3": ShapeSpec(channels=512),
    "res4": ShapeSpec(channels=1024),
    "res5": ShapeSpec(channels=2048),
}
model.neck.in_features = ["res3", "res4", "res5"]

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

N_SAMPLES_TRAIN = 4264
BATCH_SIZE = 16
EPOCH_ITERS = int(N_SAMPLES_TRAIN / BATCH_SIZE)

dataloader = get_config("common/data/odor.py").dataloader
model.num_classes = 139

train.max_iter = 50 * EPOCH_ITERS
train.output_dir = "/home/woody/iwi5/iwi5064h/backbone_ablation/rn50_in21k_timm"

train.eval_period = EPOCH_ITERS
train.checkpointer.period = 10 * EPOCH_ITERS

train.final_checkpoint = "/home/woody/iwi5/iwi5064h/backbone_ablation/rn50_in21k_timm/final"

#model.backbone.drop_path_rate = 0.4

# modify training configs
train.init_checkpoint = "/home/vault/iwi5/iwi5064h/timm_weights/resnet50_miil_21k.pth"

dataloader.train.total_batch_size = 16
dataloader.train.num_workers = 8
