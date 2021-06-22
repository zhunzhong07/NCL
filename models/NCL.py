import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
from utils.util import cluster_acc


class NCLMemory(nn.Module):
    """Memory Module for NCL"""
    def __init__(self, inputSize, K=2000, T=0.05, num_class=5, knn=None, w_pos=0.2, hard_iter=5, num_hard=400, hard_negative_start=1000):
        super(NCLMemory, self).__init__()
        self.inputSize = inputSize  # feature dim
        self.queueSize = K  # memory size
        self.T = T
        self.index = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_class = num_class
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.w_pos = w_pos
        self.hard_iter = hard_iter
        self.num_hard = num_hard
        self.hard_negative_start = hard_negative_start

        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

        self.criterion = nn.CrossEntropyLoss()
        # number of positive
        if knn == -1:
            # default set
            self.knn = int(self.queueSize / num_class / 2)
        else:
            self.knn = knn

        # label for the labeled data
        self.label = nn.Parameter(torch.zeros(self.queueSize) - 1)
        self.label.requires_grad = False

    def forward(self, q, k, labels=None, epoch=0, labeled=False, la_memory=None):
        batchSize = q.shape[0]
        self.k_no_detach = k
        k = k.detach()
        self.epoch = epoch
        self.feat = q
        self.this_labels = labels
        self.k = k.detach()
        self.la_memory = la_memory

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        x = out
        x = x.squeeze()
        if labeled:
            loss = self.supervised_loss(x, self.label, labels)
        else:
            loss = self.ncl_loss(x)

        # update memory
        self.update_memory(batchSize, q, labels)

        return loss

    def supervised_loss(self, inputs, all_labels, la_labels):
        targets_onehot = torch.zeros(inputs.size()).to(self.device)
        for i in range(inputs.size(0)):
            this_idx = all_labels == la_labels[i].float()
            one_tensor = torch.ones(1).to(self.device)
            this_idx = torch.cat((one_tensor == 1, this_idx))
            ones_mat = torch.ones(torch.nonzero(this_idx).size(0)).to(self.device)
            weights = F.softmax(ones_mat, dim=0)
            targets_onehot[i, this_idx] = weights
        # targets_onehot[:, 0] = 0.2
        targets = targets_onehot.detach()
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def ncl_loss(self, inputs):

        targets = self.smooth_hot(inputs.detach().clone())

        if self.epoch < self.hard_negative_start:

            outputs = F.log_softmax(inputs, dim=1)
            loss = - (targets * outputs)
            loss = loss.sum(dim=1)

            loss = loss.mean(dim=0)

            return loss
        else:
            loss = self.ncl_hng_loss(self.feat, inputs, targets, self.memory.clone())
            return loss

    def smooth_hot(self, inputs):
        # Sort
        value_sorted, index_sorted = torch.sort(inputs[:, :], dim=1, descending=True)
        ldb = self.w_pos
        ones_mat = torch.ones(inputs.size(0), self.knn).to(self.device)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)

        weights = F.softmax(ones_mat, dim=1) * (1 - ldb)
        targets_onehot.scatter_(1, index_sorted[:, 0:self.knn], weights)
        targets_onehot[:, 0] = float(ldb)

        return targets_onehot

    def ncl_hng_loss(self, feat, inputs, targets, memory):
        new_simi = []
        new_targets = []

        _, index_sorted_all = torch.sort(inputs[:, 1:], dim=1, descending=True)  # ignore first self-similarity
        _, index_sorted_all_all = torch.sort(inputs, dim=1, descending=True)  # consider all similarities

        if self.num_class == 5:
            num_neg = 50
        else:
            num_neg = 400

        for i in range(feat.size(0)):
            neg_idx = index_sorted_all[i, -num_neg:]
            la_memory = self.la_memory.detach().clone()
            neg_memory = memory[neg_idx].detach().clone()

            # randomly generate negative features
            new_neg_memory = []
            for j in range(self.hard_iter):
                rand_idx = torch.randperm(la_memory.size(0))
                this_new_neg_memory = (neg_memory * 1 + la_memory[rand_idx][:num_neg] * 2) / 3
                new_neg_memory.append(this_new_neg_memory)
                this_new_neg_memory = (neg_memory * 2 + la_memory[rand_idx][:num_neg] * 1) / 3
                new_neg_memory.append(this_new_neg_memory)
            new_neg_memory = torch.cat(new_neg_memory, dim=0)
            new_neg_memory = F.normalize(new_neg_memory)

            # select hard negative samples
            this_neg_simi = feat[i].view(1, -1).mm(new_neg_memory.t())
            value_sorted, index_sorted = torch.sort(this_neg_simi.view(-1), dim=-1, descending=True)
            this_neg_simi = this_neg_simi[0, index_sorted[:self.num_hard]]
            this_neg_simi = this_neg_simi / self.T

            targets_onehot = torch.zeros(this_neg_simi.size()).to(self.device)
            this_simi = torch.cat((inputs[i, index_sorted_all_all[i, :]].view(1, -1),
                                   this_neg_simi.view(1, -1)), dim=1)
            this_targets = torch.cat((targets[i, index_sorted_all_all[i, :]].view(1, -1),
                                      targets_onehot.view(1, -1)), dim=1)

            new_simi.append(this_simi)
            new_targets.append(this_targets)

        new_simi = torch.cat(new_simi, dim=0)
        new_targets = torch.cat(new_targets, dim=0)

        outputs = F.log_softmax(new_simi, dim=1)
        loss = - (new_targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)

        return loss

    def update_memory(self, batchSize, k, labels):
        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)

            if labels is not None:
                self.label.index_copy_(0, out_ids, labels.float().detach().clone())

            self.index = (self.index + batchSize) % self.queueSize
