import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
from tqdm import tqdm
from model.faster_rcnn import FasterRCNN
from dataset.dataset import VOCDataset
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou(det, gt):
  pass

def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
  pass

def load_model_and_dataset(args):
  pass

def infer(args):
  pass

def evaluate_map(args):
  pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Arguments forFaster R-CNN Inference')
  parser.add_argument('--config', dest='config_path', 
                      default='config/voc.yaml', type=str)
  parser.add_argument('--evaluate', dest='evaluate', default=False, type=bool)
  parser.add_argument('--infer_samples', dest='infer_samples',
                      default=True, type=bool)
  args = parser.parse_args()

  if args.infer_samples:
    infer(args)
  else:
    print('Not Inferring for sample as `infer_samples` argument is set to False')
  if args.evaluate:
    evaluate_map(args)
  else:
    print('Not Evaluating as `evaluate` argument is set to False')