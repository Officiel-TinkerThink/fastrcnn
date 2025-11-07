import torch
import torch.nn as nn
from src.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegionProposalNetwork(nn.Module):
  def __init__(self, in_channels=512):
    super().__init__()
    self.scales = [128, 256, 512] # this is the area of anchor
    self.aspect_ratios = [0.5, 1, 2] # this is the ratio between length and width
    self.num_anchors = len(self.scales) * len(self.aspect_ratios)

    # 3x3 conv
    self.rpn_conv = nn.Conv2d(in_channels,
                              in_channels, 
                              kernel_size=3,
                              stride=1,
                              padding=1)

    # 1x1 classification
    self.cls_layer = nn.Conv2d(in_channels,
                                self.num_anchors, # 3x3 in this case
                                kernel_size=1,
                                stride=1)
    
    # 1x1 regression
    self.bbox_reg_layer = nn.Conv2d(in_channels,
                                    self.num_anchors * 4, # 3x3x4 in this case
                                    kernel_size=1,
                                    stride=1)

  def filter_proposals(self, proposals, cls_scores, image_shape):
    """
    This would filter the proposals using nms method

    """
    
    # Pre NMS Filtering
    cls_scores = cls_scores.reshape(-1) # flatten
    cls_scores = torch.sigmoid(cls_scores) # convert logits to probability
    _, top_n_idx = cls_scores.topk(10000) # get the index of topk probability
    cls_scores = cls_scores[top_n_idx] # get the scores of that top
    proposals = proposals[top_n_idx] # get the proposals of that top, becasue the order is the same as cls_scores

    proposals = clamp_boxes_to_image_boundary(proposals, image_shape) # self-explain

    # NMS based on objectness
    keep_indices = torch.ops.torchvision.nms(proposals,
                                             cls_scores,
                                             iou_threshold=0.7)

    post_nms_keep_indices = keep_indices[
      cls_scores[keep_indices].sort(descending=True)[1]
    ]

    proposals = proposals[post_nms_keep_indices[:2000]]
    cls_scores = cls_scores[post_nms_keep_indices[:2000]]

    return proposals, cls_scores

  def generate_anchors(self, image, feat):
    """
    :param image: (B, C, H, W)  --> (1, 3, H, W)
    :param feat: (B, C, H, W)  --> (1, 512, H, W)
    :return: anchors (N, 4), anchors is with respect to image
    """
    
    grid_h, grid_w = feat.shape[-2:]  # shape is [batch_size, in_channels, grid_h, grid_w], assumption batch size is 1
    image_h, image_w = image.shape[-2:] # image is the real image, while feat is the last output feat from backbone conv

    # this below is to find the ratio of image shape to feat shape
    # so that each one stride in feat, equal to the below stride in image
    stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
    stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)

    scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
    aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)

    # the below code ensures h/w = aspect_ratios and h*w = 1
    # h * w = 1,are intended so that it keeps the area to 1 (rectangle)
    h_ratios = torch.sqrt(aspect_ratios) 
    w_ratios = 1 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).view(-1) # to get the permutation of pair
    hs = (h_ratios[:, None] * scales[None, :]).view(-1) # get permutation of pair

    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2 # creating the rectangle from w and height
    base_anchors = base_anchors.round()

    # get the shifts in x axis (0,1, ..., W_feat-1) * stride_w
    # find the shift for image
    shift_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
    shift_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h

    # find all the combination, or pairing (slide right and below)
    shifts_x, shifts_y = torch.meshgrid(shift_x, shift_y, indexing='ij')

    # (H_feat, W_feat)
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

    # create the anchors, by combining the shifts and base_anchors
    # base_anchors -> (num_anchors_per_location, 4)
    # shifts -> (H_feat * W_feat, 4)
    # the anchors -> (num_anchors per location, H_feat * W_feat, 4)
    anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
    anchors = anchors.reshape(-1, 4)

    # anchors -> (H_feat * W_feat * num_anchors_per_location, 4)
    return anchors

  def assign_targets_to_anchors(self, anchors, gt_boxes):
    """
    """
    # Get (gt_boxes, num_anchors) IOU matrix
    iou_matrix = get_iou(gt_boxes, anchors) # output shape should be (#gt, #anchors)
    # for each anchors, it select it's best match ground truth with the highest iou, the shape is M (#anchors) 
    best_match_iou, best_match_gt_index = iou_matrix.max(dim=0) # best match for each anchor with respect to gt

    # This copy will be needed later to add low
    # quality boxes
    # index range from 0 to #number of gt - 1
    best_match_gt_idx_pre_threshold = best_match_gt_index.clone()
    

    # this would classify the anchors as background, foreground or to ignore based on
    # for the best_mathc with value below 0.7 means they are not strong enough to be foreground
    below_low_threshold = best_match_iou < 0.3
    between_threshold = (best_match_iou >= 0.3) & (best_match_iou < 0.7)
    best_match_gt_index[below_low_threshold] = -1 # masking them by -1, or -2 so that they are recognized as different
    best_match_gt_index[between_threshold] = -2

    # low_quality_anchors boxes
    # get the best match for each gt, #gt
    best_anchor_iou_for_gt, _= iou_matrix.max(dim=1) # best matches for each gt
    gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])

    # get all the anchor indexes to update
    pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
    best_match_gt_index[pred_inds_to_update] = best_match_gt_idx_pre_threshold[pred_inds_to_update]

    # best match index is either valid or -1(backgroud) or -2(to ignore)
    matched_gt_boxes = gt_boxes[best_match_gt_index.clamp(min=0)]

    # set all foreground labels 1
    labels = best_match_gt_index >= 0 # this result to boolean that when convert to number become 1
    labels = labels.to(dtype=torch.float32)

    # set all background labels as 0
    background_anchors = best_match_gt_index == -1
    labels[background_anchors] = 0

    # set all ignore labels as -1
    ignored_anchors = best_match_gt_index == -2
    labels[ignored_anchors] = -1

    # Later for classification we pick labels which have >= 0
    # labels and matched_gt_boxes all have the length of #anchors
    return labels, matched_gt_boxes


  def forward(self, image, feat, target=None):
    # call RPN layers
    rpn_feat = nn.ReLU()(self.rpn_conv(feat)) # the feature map from shared conv pass to rpn conv (3x3) and pass to relu
    cls_scores = self.cls_layer(rpn_feat) # the rpn feature map pass to cls_layer (1x1) produces scores (logits)
    bbox_transform_pred = self.bbox_reg_layer(rpn_feat) # rpn passed to bbox_reg_layer produces to bbox prediction (different from anchors bounding box)

    # Generate anchors
    anchors = self.generate_anchors(image, feat)

    # cls_scores -> (Batch, Number of Anchors per location, H_feat, W_feat)
    num_of_anchors_per_location = cls_scores.size(1)
    cls_scores = cls_scores.permute(0, 2, 3, 1)
    cls_scores = cls_scores.reshape(-1, 1)
    # cls_scores -> (Batch * H_feat * W_feat * num_of_anchors_per_location, 1)

    # bbox_transform_pred -> (Batch, Number of Anchors per location * 4, H_feat, W_feat)
    bbox_transform_pred = bbox_transform_pred.permute(0, 2, 3, 1)
    bbox_transform_pred = bbox_transform_pred.view(bbox_transform_pred.size(0), 
                                                    num_of_anchors_per_location,
                                                    4, 
                                                    rpn_feat.shape[-2],
                                                    rpn_feat.shape[-1])
    bbox_transform_pred = bbox_transform_pred.permute(0, 3, 4, 1, 2)
    bbox_transform_pred = bbox_transform_pred.reshape(-1, 4)
    # bbox_transform_pred -> (Batch * H_feat * W_feat * num_of_anchors_per_location, 4)

    # Transform generated anchors according to box_transform_pred
    proposals = apply_regression_pred_to_anchors_or_proposals(
                    bbox_transform_pred.detach().reshape(-1, 1, 4), # make it three dimensions due to func requirements
                    anchors)
    #proposals -> (num_of_proposals, 4)

    proposals = proposals.reshape(proposals.size(0), 4) # make it two dimensions again
    proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape)

    rpn_output = {
      'proposals': proposals, # proposal itself is image anchor not feature
      'scores': scores,
    }

    if not self.training or target is None:
      return rpn_output
    else:
      # in training
      # Assign gt box and label for each anchor
      labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(anchors,
                                                                target['bboxes'][0])

      # Based on the gt assignment above, get regression targets for anchors
      # matched_gt_boxes_for anchors -> (Number of anchors, 4)
      # anchors -> (Number of anchors in image, 4)
      regression_targets = boxes_to_transformations_targets(
        matched_gt_boxes_for_anchors,
        anchors)

      # Sample positive and negative anchors
      sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
        labels_for_anchors,
        positive_count=128,
        total_count=256)
      sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
      localization_loss = (
        torch.nn.functional.smooth_l1_loss(
          bbox_transform_pred[sampled_pos_idx_mask],
          regression_targets[sampled_pos_idx_mask],
          beta=1/9,
          reduction='sum'
        ) / (sampled_idxs.numel())
      )

      cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_scores[sampled_idxs].flatten(),
        labels_for_anchors[sampled_idxs].flatten(),
      )

      rpn_output['rpn_classification_loss'] = cls_loss
      rpn_output['rpn_regression_loss'] = localization_loss
      return rpn_output

      

                                                        

      
    
        