import unittest
import torch

from BriPanopticSeg.Model.PanopticFpn import PanopticFPN

class TestPanopticFPN(unittest.TestCase):

    def setUp(self):
        self.model = PanopticFPN(num_thing_classes=3, num_stuff_classes=5)
        self.B, self.C, self.H, self.W = 2, 3, 512, 512
        self.images = [torch.randn(self.C, self.H, self.W) for _ in range(self.B)]

    def test_inference_output(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.images)
        self.assertIn('instances', output, "'instances' key missing in inference output")
        self.assertIn('sem_logits', output, "'sem_logits' key missing in inference output")
        self.assertEqual(
            output['sem_logits'].shape,
            (self.B, 5, self.H // 4, self.W // 4),
            f"Expected shape (B, 5, H/4, W/4), got {output['sem_logits'].shape}"
        )

    def test_training_loss_keys(self):
        self.model.train()
        targets = []
        for _ in range(self.B):
            sem_seg = torch.randint(0, 5, (self.H//4, self.W//4), dtype=torch.int64)
            boxes = torch.tensor([[10, 10, 100, 100]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
            masks = torch.randint(0, 2, (1, self.H, self.W), dtype=torch.uint8)
            targets.append({
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "sem_seg": sem_seg
            })

        loss_dict = self.model(self.images, targets)
        self.assertIn('loss_sem_seg', loss_dict, "'loss_sem_seg' key missing in training loss")
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in loss_dict.values()), "All values should be tensors")

if __name__ == '__main__':
    unittest.main()

