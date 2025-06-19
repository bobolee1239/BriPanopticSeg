import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from typing import List, Dict, Optional, Union


class SemanticHead(nn.Module):
    def __init__(self, in_channels_list: List[int], num_classes: int) -> None:
        super().__init__()
        self.inner_blocks = nn.ModuleDict()
        self.layer_blocks = nn.ModuleDict()

        # FPN levels: P2, P3, P4, P5
        # Corresponding spatial resolution relative to original image (H, W):
        #   P2: H/4,  W/4
        #   P3: H/8,  W/8
        #   P4: H/16, W/16
        #   P5: H/32, W/32
        for idx, level in enumerate(['0', '1', '2', '3']):
            self.inner_blocks[level] = nn.Conv2d(in_channels_list[idx], 128, kernel_size=1)
            self.layer_blocks[level] = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.output_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, features: Dict[str, torch.Tensor], image_shape: List[int]) -> torch.Tensor:
        # Start from the lowest resolution (P5) and upsample step-by-step
        x = self.inner_blocks['3'](features['3'])
        for level in ['2', '1', '0']:
            inner_lateral = self.inner_blocks[level](features[level])
            x = F.interpolate(x, size=inner_lateral.shape[-2:], mode='nearest') + inner_lateral
        x = self.layer_blocks['0'](x)  # refine with conv

        # Upsample to original resolution
        x = F.interpolate(x, size=image_shape, mode='bilinear', align_corners=False)
        return self.output_head(x)

class PanopticFPN(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.backbone = resnet_fpn_backbone(
            'resnet50',
            weights=ResNet50_Weights.DEFAULT  # modern and future-proof
        )
        # Input: [B, 3, H, W] -> Output: Dict[str, Tensor] with:
        #   "0" (P2):   [B, 256, H/4,  W/4]
        #   "1" (P3):   [B, 256, H/8,  W/8]
        #   "2" (P4):   [B, 256, H/16, W/16]
        #   "3" (P5):   [B, 256, H/32, W/32]
        #   "pool" (P6):[B, 256, H/64, W/64]

        self.mask_rcnn = MaskRCNN(
            self.backbone,
            num_classes=num_classes,
            min_size=512,
            max_size=1024
        )

        in_channels_list = [256, 512, 1024, 2048]  # channels for P2-P5 from ResNet50
        self.semantic_head = SemanticHead(in_channels_list=in_channels_list, num_classes=num_classes)
        return

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None
                ) -> Union[Dict[str, torch.Tensor], Dict[str, Union[List, torch.Tensor]]]:
        if self.training:
            return self.forward_train(images, targets)
        else:
            return self.forward_infer(images)

    def forward_train(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        features = self.backbone.body(torch.stack(images))
        # [B, 3, H, W] -> FPN features

        image_shape = images[0].shape[-2:]
        sem_logits = self.semantic_head(features, image_shape)
        # [B, num_stuff_classes, H/4, W/4]

        sem_gt_list = [t['sem_seg'] for t in targets]
        # list of [H, W] int64 label maps

        sem_gt_tensor = torch.stack(sem_gt_list).long()
        loss_sem_seg = F.cross_entropy(sem_logits, sem_gt_tensor)
        # scalar loss

        loss_dict = self.mask_rcnn(images, targets)
        # Dict[str, Tensor] of losses from instance head

        _lambda = 0.5
        loss_dict['loss_sem_seg'] = _lambda*loss_sem_seg
        return loss_dict

    def forward_infer(self, images: List[torch.Tensor]) -> Dict[str, Union[List, torch.Tensor]]:
        with torch.no_grad():
            features = self.backbone.body(torch.stack(images))
            # [B, 3, H, W] -> FPN features

            image_shape = images[0].shape[-2:]
            sem_logits = self.semantic_head(features, image_shape)
            # [B, num_stuff_classes, H, W]

            instance_preds = self.mask_rcnn(images)
            # List[Dict[str, Tensor]]

        return {
            'instances': instance_preds,  # List[Dict[str, Tensor]]
            'sem_logits': sem_logits      # [B, num_stuff_classes, H/4, W/4]
        }
