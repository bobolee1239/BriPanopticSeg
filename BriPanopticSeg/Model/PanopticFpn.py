import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from typing import List, Dict, Optional, Union


class SemanticHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            # [B, C, H/4, W/4] -> [B, 256, H/4, W/4]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
            # [B, 256, H/4, W/4] -> [B, num_classes, H/4, W/4]
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = features['0']
        # [B, C, H/4, W/4] from P2 level of FPN (highest resolution)
        return self.head(x)
        # [B, num_classes, H/4, W/4]

class PanopticFPN(nn.Module):
    def __init__(
        self,
        num_thing_classes: int,
        num_stuff_classes: int
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
            num_classes=num_thing_classes,
            min_size=512,
            max_size=1024
        )

        self.semantic_head = SemanticHead(in_channels=256, num_classes=num_stuff_classes)

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None
                ) -> Union[Dict[str, torch.Tensor], Dict[str, Union[List, torch.Tensor]]]:
        if self.training:
            return self.forward_train(images, targets)
        else:
            return self.forward_infer(images)

    def forward_train(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        features = self.backbone.body(torch.stack(images))
        # [B, 3, H, W] -> FPN features

        sem_logits = self.semantic_head(features)
        # [B, num_stuff_classes, H/4, W/4]

        sem_gt_list = [t['sem_seg'] for t in targets]
        # list of [H, W] int64 label maps

        sem_gt_tensor = torch.stack(sem_gt_list).long()
        # [B, H, W]

        loss_sem_seg = F.cross_entropy(sem_logits, sem_gt_tensor)
        # scalar loss

        loss_dict = self.mask_rcnn(images, targets)
        # Dict[str, Tensor] of losses from instance head

        loss_dict['loss_sem_seg'] = loss_sem_seg
        return loss_dict

    def forward_infer(self, images: List[torch.Tensor]) -> Dict[str, Union[List, torch.Tensor]]:
        with torch.no_grad():
            features = self.backbone.body(torch.stack(images))
            # [B, 3, H, W] -> FPN features

            sem_logits = self.semantic_head(features)
            # [B, num_stuff_classes, H/4, W/4]

            instance_preds = self.mask_rcnn(images)
            # List[Dict[str, Tensor]]

        return {
            'instances': instance_preds,  # List[Dict[str, Tensor]]
            'sem_logits': sem_logits      # [B, num_stuff_classes, H/4, W/4]
        }
