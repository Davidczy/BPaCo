import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.0):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt

        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth

        self.weight = None

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        if self.weight is not None:
            anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)
        else:
            anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = self.smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000):
        super(MultiTaskLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999


    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        rew = self.class_weight.squeeze()[labels[:batch_size].squeeze()]
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos * rew
        loss = loss.mean()

        loss_balancesoftmax = F.cross_entropy(sup_logits+self.weight, labels[:batch_size].squeeze())
        return loss_balancesoftmax + self.alpha * loss


class MultiTaskBLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000):
        super(MultiTaskBLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999


    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        loss_ce = F.cross_entropy(sup_logits, labels[:batch_size].squeeze())
        return loss_ce + self.alpha * loss


class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets):

        device = (torch.device('cuda')  
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(len(self.cls_num_list))[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:2 * batch_size].mm(features.T)

        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss



class BPaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.0, cls_num_list1=None):
        super(BPaCoLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt

        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth

        self.weight = None
        self.cls_num_list1 = cls_num_list1

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))
    
    def forward(self, features, labels=None, sup_logits=None, centers=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        labels_centers = torch.arange(len(self.cls_num_list1), device=device).view(-1, 1)
        labels1 = torch.cat([labels[:batch_size], labels_centers], dim=0)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
        mask1 = torch.eq(labels1[:batch_size], labels1.T).float().to(device)
        batch_cls_count = torch.eye(len(self.cls_num_list1), device=device)[labels1].sum(dim=0).squeeze()

        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        if self.weight is not None:
            anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)
        else:
            anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask


        logits_mask1 = torch.scatter(
            torch.ones_like(mask1),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask1 = mask1 * logits_mask1
        
        # Balanced Process
        features1 = torch.cat([features[:batch_size], centers], dim=0)
        logits1 = features1[:batch_size].mm(features1.T)
        logits1 = torch.div(logits1, self.temperature)
        logits_max1, _ = torch.max(logits1, dim=1, keepdim=True)
        logits1 = logits1 - logits_max1.detach()
        # class-averaging
        exp_logits1 = torch.exp(logits1) * logits_mask1
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in labels1], device=device).view(1, -1).expand(
            batch_size, batch_size + len(self.cls_num_list1)) - mask1
        exp_logits_sum1 = exp_logits1.div(per_ins_weight).sum(dim=1, keepdim=True)
        log_prob1 = logits1 - torch.log(exp_logits_sum1)
        mean_log_prob_pos1 = (mask1 * log_prob1).sum(1) / mask1.sum(1)
        loss1 = - mean_log_prob_pos1
        loss1 = loss1.mean()


        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = self.smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        loss = loss1 + loss
        return loss
    


class BPaCoplusLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=8192, num_classes=1000, smooth=0.0, cls_num_list1=None):
        super(BPaCoplusLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt

        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth

        self.weight = None
        self.cls_num_list1 = cls_num_list1

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))
    
    def forward(self, features, labels=None, sup_logits=None, centers=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        labels_centers = torch.arange(len(self.cls_num_list1), device=device).view(-1, 1)
        labels1 = torch.cat([labels[:batch_size], labels_centers], dim=0)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
        mask1 = torch.eq(labels1[:batch_size], labels1.T).float().to(device)
        batch_cls_count = torch.eye(len(self.cls_num_list1))[labels].sum(dim=0).squeeze()
        batch_cls_count1 = torch.eye(len(self.cls_num_list1))[labels1].sum(dim=0).squeeze()

        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        if self.weight is not None:
            anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)
        else:
            anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask


        logits_mask1 = torch.scatter(
            torch.ones_like(mask1),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask1 = mask1 * logits_mask1
        
        # Balanced Process
        features1 = torch.cat([features[:batch_size], centers], dim=0)
        logits1 = features1[:batch_size].mm(features1.T)
        logits1 = torch.div(logits1, self.temperature)
        logits_max1, _ = torch.max(logits1, dim=1, keepdim=True)
        logits1 = logits1 - logits_max1.detach()
        # class-averaging
        exp_logits1 = torch.exp(logits1) * logits_mask1
        per_ins_weight1 = torch.tensor([batch_cls_count1[i] for i in labels1], device=device).view(1, -1).expand(
            batch_size, batch_size + len(self.cls_num_list1)) - mask1
        exp_logits_sum1 = exp_logits1.div(per_ins_weight1).sum(dim=1, keepdim=True)
        log_prob1 = logits1 - torch.log(exp_logits_sum1 +1e-12)
        mean_log_prob_pos1 = (mask1 * log_prob1).sum(1) / mask1.sum(1)
        loss1 = - mean_log_prob_pos1
        loss1 = loss1.mean()


        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = self.smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        # print(sup_logits.shape)
        # print(labels.shape)
        # print(torch.tensor([batch_cls_count[i] for i in labels], device=device).shape)
        # print(exp_logits.shape)
        # print(mask.shape)
        # per_ins_weight = torch.tensor([batch_cls_count[i] for i in labels], device=device).view(1, -1).expand(
        #     batch_size, labels.shape[0])
        # per_ins_weight = torch.cat([sup_logits, per_ins_weight], dim=1) - mask
        labels = torch.cat([labels_centers, labels], dim=0)
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in labels], device=device).view(1, -1).expand(
            batch_size,  labels.shape[0]) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        loss = loss1 + loss
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss