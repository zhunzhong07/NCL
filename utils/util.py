from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
# from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.optimize import linear_sum_assignment
import random
import os
import argparse
#######################################################
# Evaluate Critiron
#######################################################

def cluster_acc(y_true, y_pred):
  """Returns the clustering accuracy.

  Adapted from https://github.com/k-han/AutoNovel
  The output of `linear_sum_assignment` is transposed as explained in:
  https://stackoverflow.com/a/57992848/2006462

  Args:
    y_true: The ground truth label.
    y_pred: The predicted label.

  Returns:
    The clustering accuracy.
  """
  y_true = y_true.astype(np.int64)
  assert y_pred.size == y_true.size
  d = max(y_pred.max(), y_true.max()) + 1
  w = np.zeros((d, d), dtype=np.int64)
  for i in range(y_pred.size):
    w[y_pred[i], y_true[i]] += 1
  ind = linear_sum_assignment(w.max() - w)
  ind = np.asarray(ind)
  ind = np.transpose(ind)
  return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc_old(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class BCE_softlabels(nn.Module):
    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)

        P = prob1.mul_(prob2)
        P = P.sum(1)
        neglogP = - (simi * torch.log(P + BCE.eps) + (1. - simi) * torch.log(1. - P + BCE.eps))
        return neglogP.mean()

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CentroidManager:
  """Manages the lifecycle of class prototypes (centroids).

  Attributes:
    centroids: The prototypes vectors.
    enable_alignment_loss: Controls when to start alignment.
    current_batch_centroids: The prototypes of classes in the current batch.
    num_classes: Number of classes to expect.
    predictions: Class assignment.
    count: Number of times instances of a class is met.
    enable_mds_loss: Whether to enable MDS loss or not.
    equi_dist_centroids: The equi-distant centroids.
    max_dist: The max distance between the initial centroids.
  """

  def __init__(self):
    self.centroids = []
    self.enable_alignment_loss = False
    self.current_batch_centroids = []
    self.equi_dist_centroids = []
    self.max_dist = 0

  def get_init_status(self):
    if self.enable_alignment_loss:
      return True
    else:
      return False

  def enable_loss(self):
    self.enable_alignment_loss = True

  def disable_loss(self):
    self.enable_alignment_loss = False

  def initialize(self, features, num_classes, enable_mds_loss=True):
    """Initialize the centroids with the current features."""
    self.num_classes = num_classes
    self.enable_mds_loss = enable_mds_loss

    km_model = KMeans(n_clusters=self.num_classes, n_init=20)

    self.predictions = km_model.fit_predict(features)
    self.centroids = km_model.cluster_centers_
    self.count = 100 * np.ones(self.num_classes, dtype=np.int)

    if self.enable_mds_loss:
      self.prepare_equi_dist_centroids(features)

  def prepare_equi_dist_centroids(self, features):
    """Initialize equi distant centroids with MDS."""
    distance_weight = 1
    self.max_dist = distance_weight * distance.pdist(self.centroids).max()
    distances = self.max_dist * np.ones((self.num_classes, self.num_classes))
    np.fill_diagonal(distances, 0)
    print(np.array(features).shape)
    z_dim = np.array(features).shape[1]
    self.equi_dist_centroids = MDS(
        n_components=z_dim,
        dissimilarity='precomputed').fit(distances).embedding_

  def prepare_targets(self, start_index, end_index):
    """Prepare targets for each mini-batch."""
    preds = self.predictions[start_index:end_index]
    self.current_batch_centroids = self.centroids[preds]
    return self.current_batch_centroids

  def update_centroids(self, features, start_index, end_index):
    """Update centroids with each mini-batch features.

    It involves two steps:
      1) reassignment of the current pseudo labels.
      2) Updating the class prototypes with the new assignment.

    Args:
      features: mini-batch features.
      start_index: Starting index of mini-batch
      end_index: Ending index of mini-batch
    """
    dist = distance.cdist(features, self.centroids)
    new_assignments = np.argmin(dist, axis=1)
    self.predictions[start_index:end_index] = new_assignments

    for index in range(features.shape[0]):
      class_id = new_assignments[index]
      self.count[class_id] += 1
      eta = 1.0 / self.count[class_id]
      if len(self.equi_dist_centroids) > 0:
        self.centroids[class_id] = self.centroids[class_id] - eta * (
            self.centroids[class_id] -
            (features[index] + self.equi_dist_centroids[class_id]))
      else:
        self.centroids[class_id] = self.centroids[class_id] + eta * (
            self.centroids[class_id] + features[index])
        # self.centroids[class_id] = (
        #     1 - eta) * self.centroids[class_id] + eta * features[index]

    print('Class count: {}'.format(self.count))

  def reinit_centroids(self, features):
    km_model = KMeans(
        n_clusters=self.num_classes, init=self.centroids, n_init=1)
    self.predictions = km_model.fit_predict(features)
    self.centroids = km_model.cluster_centers_
    # if self.enable_mds_loss:
    #   self.prepare_equi_dist_centroids(features)

  def get(self):
    return self.centroids

  def get_labels(self, features):
    dist = distance.cdist(features, self.centroids)
    labels = np.argmin(dist, axis=1)
    return labels