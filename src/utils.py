import torch

def apply_regression_pred_to_anchors_or_proposals(
  box_transform_pred, anchors_or_proposals):
  """
  :param box_transform_pred: (num_anchors_or_proposal, num_classes, 4)
  :param anchors_or_proposals: (num_anchors_or_proposal, 4)
  :return: pred_boxes: (num_anchors_or_proposal, num_classes, 4)
  """
  box_tranform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)

  # get xs, cy, w, h from x1, y1, x2, y2 of anchors
  w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
  h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
  center_x = anchors_or_proposals[:, 0] + 0.5 * w
  center_y = anchors_or_proposals[:, 1] + 0.5 * h

  # get the dx, dy, dw, dh from box_pred (this is not true dimension, but rather transformation coefficient)
  dx = box_tranform_pred[:, 0]
  dy = box_tranform_pred[:, 1]
  dw = box_tranform_pred[:, 2]
  dh = box_tranform_pred[:, 3]
  # dh -> (num_anchors_or_proposals, num_classes)

  # this predict center and w, h of the bbox of image
  pred_center_x = dx * w[:, None] + center_x[:, None]
  pred_center_y = dy * h[:, None] + center_y[:, None]
  pred_w = torch.exp(dw) * w[:, None]
  pred_h = torch.exp(dh) * h[:, None]
  # pred_center_x -> (num_anchors_or_proposals, num_classes)

  # convert that to x1,y1, x2, y2 format
  pred_boxes_x1 = pred_center_x - 0.5 * pred_w
  pred_boxes_y1 = pred_center_y - 0.5 * pred_h
  pred_boxes_x2 = pred_center_x + 0.5 * pred_w
  pred_boxes_y2 = pred_center_y + 0.5 * pred_h

  pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=-1)
  return pred_boxes

def get_iou(boxes1, boxes2):
  """
  :param boxes1: (N, 4)
  :param boxes2: (M, 4)
  :return: iou: (N, M)
  """
  # Area of boxes (x2-x1) * (y2-y1)
  area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
  area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

  # Intersection
  x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0]) # (N, M)
  y_left = torch.max(boxes1[:, None, 1], boxes2[:, 1]) # (N, M)
  x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2]) # (N, M)
  y_right = torch.min(boxes1[:, None, 3], boxes2[:, 3]) # (N, M)

  intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_right - y_left, min=0)
  union_area = area1[:, None] + area2 - intersection_area
  return intersection_area / union_area # (N, M)

def clamp_boxes_to_image_boundary(boxes, image_shape):
  boxes_x1 = boxes[..., 0]
  boxes_y1 = boxes[..., 1]
  boxes_x2 = boxes[..., 2]
  boxes_y2 = boxes[..., 3]
  height, width = image_shape[-2:]

  boxes_x1 = torch.clamp(boxes_x1, min=0, max=width)
  boxes_y1 = torch.clamp(boxes_y1, min=0, max=height)
  boxes_x2 = torch.clamp(boxes_x2, min=0, max=width)
  boxes_y2 = torch.clamp(boxes_y2, min=0, max=height)

  boxes = torch.cat((
    boxes_x1[..., None],
    boxes_y1[..., None],
    boxes_x2[..., None],
    boxes_y2[..., None]
  ), dim=-1)

  return boxes

def boxes_to_transformations_targets(ground_truth_boxes, anchors_or_proposals):
  # Get center_x, center_y, w, h from x1, x2, y1, y2 for anchors
  widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
  heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
  center_x = anchors_or_proposals[:, 0] + 0.5 * widths
  center_y = anchors_or_proposals[:, 1] + 0.5 * heights 

  # Get center_x, center_y, w, h from x1, x2, y1, y2 for gt_boxes
  gt_widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
  gt_heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
  gt_center_x = anchors_or_proposals[:, 0] + 0.5 * widths
  gt_center_y = anchors_or_proposals[:, 1] + 0.5 * heights

  # calculate the target transformaton parameters for each anchors
  target_dx = (gt_center_x - center_x) / widths
  target_dy = (gt_center_y - center_y) / heights
  target_dw = torch.log(gt_widths / widths)
  target_dh = torch.log(gt_heights / heights)

  # combine them into one union with second axis as stack axis
  regression_targets = torch.stack(
    (target_dx, 
    target_dy,
    target_dw,
    target_dh), dim=1)

  return regression_targets


def sample_positive_negative(labels, positive_count, total_count):
  positive = torch.where(labels >= 1)[0] # this return idx with positive labels
  negative = torch.where(labels == 0)[0] # this return idx with negative labels

  num_pos = positive_count
  num_pos = min(positive.numel(), num_pos)
  num_neg = total_count - num_pos
  num_neg = min(negative.numel(), num_neg)
  perm_positive_idxs = torch.randperm(positive.numel(),
    device=positive.device)[:num_pos] # return idx relative to positive array (which itself idx of labels)
  perm_negative_idxs = torch.randperm(negative.numel(),
    device=negative.device)[:num_neg] # return idx relative to negative array (which itself idx of labels)
  pos_idxs = positive[perm_positive_idxs] # idx of labels which positive
  neg_idxs = negative[perm_negative_idxs] # idx of labels which negative
  sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool) # placeholder for positives
  sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool) # placeholder for negatives
  sampled_pos_idx_mask[pos_idxs] = True # labels with positive criteria sampled assigned as true
  sampled_neg_idx_mask[neg_idxs] = True # labels with negative criteria sampled assigned as false
  return sampled_neg_idx_mask, sampled_pos_idx_mask

def transform_boxes_to_original_size(boxes, new_size, original_size):
    ratios = [
      torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) / torch.tensor(s, dtype=torch.float32, device=boxes.device)
      for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    ymin = ymin * ratio_height
    xmax = xmax * ratio_width
    ymax = ymax * ratio_height
    boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
    return boxes