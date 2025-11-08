import torch
import torch.nn as nn
from .rpn import RegionProposalNetwork
from .roi import ROIHead
import torchvision
from src.utils import transform_boxes_to_original_size

class FasterRCNN(nn.Module):
  def __init__(self, num_classes, model_config):
    super().__init__()
    vgg16 = torchvision.models.vgg16(pretrained=True)
    self.backbone = vgg16.features[:-1]
    self.rpn = RegionProposalNetwork(model_config['backbone_out_channels'],
                                      scales=model_config['scales'],
                                      aspect_ratios=model_config['aspect_ratios'],
                                      model_config=model_config)
    self.roi_head = ROIHead(model_config, num_classes, in_channels=model_config['backbone_out_channels'])
    for layer in self.backbone[:10]: # freeze first 10 layers, the rest is trainable
      for p in layer.parameters():
        p.requires_grad = False
    
    self.image_mean = model_config['image_mean']
    self.image_std = model_config['image_std']
    self.min_size = model_config["min_im_size"]
    self.max_size = model_config["max_im_size"]

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
    # the below is to scaling the image into the range of [600 for shorter side, 1000 for longer side]
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
    frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)

    if not self.training:
      # transform boxes to original image dimension
      frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'],
        image.shape[-2:], 
        old_shape
      )
      
    return rpn_output, frcnn_output

