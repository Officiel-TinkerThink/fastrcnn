import torch
import torch.nn as nn
from src.utils import *

class ROIHead(nn.Module):
  """
  it's important to note that the feature that pass here is not the overall feat of image, but only roi.
  """
  def __init__(self, num_classes=21, n_channels=512):
    super().__init__()
    self.num_classes = num_classes
    self.pool_size=7
    self.fc_inner_dim=1024
    # first layer that translates from feat (rectangle) to 1d array
    self.fc6 = nn.Linear(n_channels * self.pool_size * self.pool_size, self.fc_inner_dim)

    # second layer that translates 1d array to 1d array
    self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)

    # layer that classify
    self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)

    # layer that regression
    self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)

  def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
    # Get (gt_boxes, num_proposals) IOU matrix
    iou_matrix = get_iou(gt_boxes, proposals) # output should be #gt, #proposals, 
    best_match_iou, best_match_gt_index = iou_matrix.max(dim=0) # best match for each proposal with respect to gt

    # this would classify the proposals as background, foreground based on
    # if there is a match with any of the gt boxes
    below_low_threshold = best_match_iou < 0.5
    best_match_gt_index[below_low_threshold] = -1
    # best match index is either valid or -1(backgroud)
    matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_index.clamp(min=0)]

    labels = gt_labels[best_match_gt_index.clamp(min=0)]
    labels = labels.to(dtype=torch.int64)

    # set all background labels as 0
    background_proposals = best_match_gt_index == -1
    labels[background_proposals] = 0

    # Later for classification we pick labels which have >= 0
    return labels, matched_gt_boxes_for_proposals

  def filter_prediction(self, pred_boxes, pred_labels, pred_scores):
    # remove low scoring boxes
    keep = torch.where(pred_scores > 0.05)[0]
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

    # remove small boxes
    min_size = 1
    ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
    keep = torch.where((ws >= min_size) & (hs >= min_size))[0]
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

    # class wise nms
    keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
    for class_id in torch.unique(pred_labels):
      curr_indices = torch.where(pred_labels == class_id)[0]
      curr_keep_indices = torch.ops.torchvision.nms(
        pred_boxes[curr_indices],
        pred_scores[curr_indices],
        iou_threshold=0.5
      )
      keep_mask[curr_indices[curr_keep_indices]] = True
    
    keep_indices = torch.where(keep_mask)[0]
    post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
    keep = post_nms_keep_indices[:100]
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

    return pred_boxes, pred_scores, pred_labels

  def forward(self, feat, proposals, image_shape, target):
    if self.training and target is not None:
      gt_boxes = target['bboxes'][0]
      gt_labels = target['labels'][0]
      # assign labels and gt boxes for proposals
      # Assign gt box and label for each proposal
      labels, matched_gt_boxes_for_proposals = self.assign_targets_to_proposals(
                                                                                proposals,
                                                                                gt_boxes, gt_labels)

      # Based on the gt assignment above, get regression targets for proposals
      # matched_gt_boxes_for proposals -> (Number of proposals, 4)
      # proposals -> (Number of proposals in image, 4)
      

      # Sample positive and negative proposals
      sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
        labels,
        positive_count=128,
        total_count=256)
      sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

      proposals = proposals[sampled_idxs]
      labels = labels[sampled_idxs]

      matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]

      regression_targets = boxes_to_transformations_targets(
        matched_gt_boxes_for_proposals,
        proposals)
      # regression_targets -> (sampled_training_proposals, 4)

    # ROI pooling part
    # spatial scale for roi pooling
    # for vgg16 this would be 1/26
    spatial_scale = 0.0625

    proposal_roi_pool_feats = torchvision.ops.roi_pool(
      feat,
      [proposals],
      output_size=self.pool_size,
      spatial_scale=spatial_scale
    )

    proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
    box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
    box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
    cls_scores = self.cls_layer(box_fc_7)
    bbox_transform_pred = self.bbox_reg_layer(box_fc_7)

    num_boxes, num_classes = cls_scores.shape
    box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)

    frcnn_output = {}

    if self.training and target is not None:
      classification_loss = torch.nn.functional.cross_entropy(
        cls_scores,
        labels
      )
      # Compute localization only for the non-background
      fg_proposals_idxs = torch.where(labels > 0)[0]
      # get class labels for them
      fg_class_labels = labels[fg_proposals_idxs]

      localization_loss = torch.nn.functional.smooth_l1_loss(
        bbox_transform_pred[fg_proposals_idxs, fg_class_labels],
        regression_targets[fg_proposals_idxs],
        beta=1/9,
        reduction='sum'
      ) / labels.numel()
      frcnn_output['frcnn_classification_loss'] = classification_loss
      frcnn_output['frcnn_localization_loss'] = localization_loss
      return frcnn_output
    else:
      # Aplly tranformation predictions to proposals
      pred_boxes = apply_regression_pred_to_anchors_or_proposals(
        bbox_transform_pred,
        proposals
      )
      pred_scores = torch.nn.functional.softmax(cls_scores, dim=1)

      # clamp boxes to image boundary
      pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)

      # create labels for each prediction
      pred_labels = torch.arange(num_classes, device=cls_scores.device)
      pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

      # remove background class predictions
      pred_boxes = pred_boxes[:, 1:]
      pred_scores = pred_scores[:, 1:]
      pred_labels = pred_labels[:, 1:]
      # pred_boxes -> (num_proposals, num_classes-1, 4)

      pred_boxes = pred_boxes.reshape(-1, 4)
      pred_scores = pred_scores.reshape(-1)
      pred_labels = pred_labels.reshape(-1)

      # filter proposals
      pred_boxes, pred_scores, pred_labels = self.filter_prediction(pred_boxes,
                                                                    pred_labels, 
                                                                    pred_scores)
      frcnn_output['pred_boxes'] = pred_boxes
      frcnn_output['pred_scores'] = pred_scores
      frcnn_output['pred_labels'] = pred_labels
      return frcnn_output


