import torch
import torch.nn as nn
from .rpn import RegionProposalNetwork
from .roi import ROIHead
import torchvision
from src.utils import transform_boxes_to_original_size

class FasterRCNN(nn.Module):
  def __init__(self, num_classes=21):
    super().__init__()
    vgg16 = torchvision.models.vgg16(pretrained=True)
    self.backbone = vgg16.features[:-1]
    self.rpn = RegionProposalNetwork()
    self.roi_head = ROIHead(num_classes, in_channels=512)
    for layer in slef.backbone[:10]:
      for p in layer.parameters():
        p.requires_grad = False
    
    self.image_mean = [0.485, 0.456, 0.406]
    self.image_std = [0.229, 0.224, 0.225]
    self.min_size = 600
    self.max_size = 1000

  def normalize_resize_image_and_boxes(self, image, bboxes):
    # normalize
    mean = torch.as_tensor(self.image_mean, dtype=image.dtype, device=image.device)
    std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
    image = (image - mean[:, None, None]) / std[:, None, None]

    # resize such that lower dim is scaled to 600
    # and upper dim is scaled to 1000
    h, w = image.shape[-2:]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
    scale_factor = scale.item()
    # Resize image based on scale
    image = torch.nn.functional.interpolate(
      image,
      size=None,
      scale_factor=scale_factor,
      mode='bilinear',
      recompute_scale_factor=True,
      align_corners=False
    )

    # Resize bboxes
    if bboxes is not None:
      ratios = [
        torch.tensor(s, dtype=torch.float, device=bboxes.device) / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
        for s, s_orig in zip(image.shape[-2:], (h,w))
      ]
      ratio_height, ratio_width = ratios
      xmin, ymin, xmax, ymax = bboxes.unbind(2)
      xmin = xmin * ratio_width
      ymin = ymin * ratio_height
      xmax = xmax * ratio_width
      ymax = ymax * ratio_height
      bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
    return image, bboxes


  def forward(self, image, target=None):
    old_shape = image.shape[-2:]
    if self.training:
      # Normalize and resize boxes
      image, bboxes = self.normalize_resize_image_and_boxes(
        image, target['bboxes']
      )
      target['bboxes'] = bboxes
    else:
      image, _ = self.normalize_resize_image_and_boxes(
        image, None
      )

      # call backbone
      feat = self.backbone(image)

      # call RPN and get proposals
      rpn_output = self.rpn(image, feat, target)
      proposals = rpn_output['proposals']

      # Call ROI head and convert proposals to boxes
      frcnn_output = self.roi_head(feat, proposals, image.shape[:2:])

      if not self.training:
        # transform boxes to original image dimension
        frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'],
          image.shape[-2:], 
          old_shape
        )
      
      return rpn_output, frcnn_output

