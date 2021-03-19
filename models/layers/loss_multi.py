import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
#import matplotlib.pyplot as plt
import numpy as np
from itertools import repeat
from models.layers.loss_correlation import ncc_loss

from models.layers.loss_hausdorff import av_hausdorff_dist_torch
from models.surface_loss.distances import HDDTBinaryLoss

def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

# OS: Hausdorff distance loss on the second derivative of both the target and input
class hausdorff_loss(nn.Module):
    def __init__(self, n_classes=3):
        super(hausdorff_loss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward

        self.n_classes = n_classes

        device = "cuda" #"cuda:0"
        type = torch.float32
        # self.lapKernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        #                                [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
        #                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
        #                               device=device, dtype=type, requires_grad=False)

        self.lapKernel = torch.tensor([[[2/26, 3/26, 2/26], [3/26, 6/26, 3/26], [2/26, 3/26, 2/26]],
                                       [[3/26, 6/26, 3/26], [6/26, -88/26, 6/26], [3/26, 6/26, 3/26]],
                                       [[2/26, 3/26, 2/26], [3/26, 6/26, 3/26], [2/26, 3/26, 2/26]]],
                                       device=device, dtype=type, requires_grad=False)

        self.tmp = []
        self.tmp_input = []
    def forward(self, input, target, weight=None, size_average=True):
        n, c, h, w, s = input.size()

        # Apply 3D convolution with a laplacian kernel
        input = F.softmax(input, dim=1)

        weights = self.lapKernel.view(1, 1, 3, 3, 3).repeat(3, c, 1, 1, 1)
        #input = F.conv3d(input, weights)#.view(n, self.n_classes, -1)

        weights2 = self.lapKernel.view(1, 1, 3, 3, 3).repeat(1, 1, 1, 1, 1)

        target = self.one_hot_encoder(target).contiguous()
        for ii in range(c):
            self.tmp_input.append(F.conv3d(input[:, ii, :, :, :].unsqueeze(1), weights2))
            self.tmp.append(F.conv3d(target[:, ii, :, :, :].unsqueeze(1), weights2))#.squeeze(1))
        input = torch.cat(self.tmp_input, axis = 1)#.view(n, self.n_classes, -1)
        target = torch.cat(self.tmp, axis = 1)#.view(n, self.n_classes, -1)

        # delt = 0.00001
        # target[target < delt] = -10

        tmp_target = target[:, 0, ...].squeeze()
        #tar_x = tmp_target[]
        #target_points = torch.cat((), dim=1)

        loss = -1

        return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.00001 #0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        #input = input.view(batch_size, self.n_classes, -1) # Shouldn't this be used instead?
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        #union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        # This is the proper smoothing, as we multiply inter by 2 in the next line. This way, the loss is between [0, 1]
        union = torch.sum(input, 2) + torch.sum(target, 2) + 2.0 * smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

# OS: my addition - Tversky loss
class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=False, device="cuda")
        self.beta = torch.tensor(beta, dtype=torch.float32, requires_grad=False, device="cuda")
        # TODO: maybe add different alpha and beta per each class?

    def forward(self, rank, input, target):
        smooth = 0.00001 #0.01
        batch_size = input.size(0)

        if len(self.alpha) > 1:
            alpha = self.alpha.repeat(batch_size, 1).to(rank)
        else:
            alpha = self.alpha

        if len(self.beta) > 1:
            beta = self.beta.repeat(batch_size, 1).to(rank)
        else:
            beta = self.beta

        # NOTE: Not sure why we need the softmax
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        #input = input.view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(rank, target).contiguous().view(batch_size, self.n_classes, -1)

        tp = torch.sum(input * target, 2) + smooth # True positives
        fp = torch.sum(input * (1.0 - target), 2) # False positive
        fn = torch.sum((1.0 - input) * target, 2) # False negative

        # Compute the Tversky loss (this is elementwise division, and then we sum over all categories and batch size)
        tmp = tp / (tp + alpha * fp + beta * fn)
        #tmp[:, 0] = 0.05 * tmp[:, 0] # The first class is background, which has ~100 times more pixels than the other classes
        score = torch.sum(tmp)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes)) # Normalize

        return score

class jointloss_multi(nn.Module):
    def __init__(self, n_classes, alpha=0.5, beta=0.5, lam=1, hausdorff=-1, rank=0):
        super(jointloss_multi, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.diceloss = SoftDiceLoss(n_classes=n_classes)
        self.tvloss = TverskyLoss(n_classes=n_classes, alpha=alpha, beta=beta)
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        #self.avHaus = av_hausdorff_dist_torch

        # TODO: can be extended to lam_i, i = 1,...,n_classes
        self.lam = lam # lam controls the relative weight between the tversky similarity and overlap minimization

        # Hausdorff distance - negative value for hausdorff to remove this metric from training
        # Dafault behavior is not to minimize the hausdorff distance, due to high computational complexity
        self.lambda_haus = hausdorff
        if hausdorff > 0:
            self.crsEnt = HDDTBinaryLoss()

        # TODO: Make this external
        self.haus_mult_factor = 5 #10
        self.haus_option = 0 # 0 - Hausdorff distance for the entire GP organ, 1 - for each side and GPi/e separately, 2 - for each label separately, no left/right

    def forward(self, rank, input, target, epoch_cnt):
        # Calculate the multiclass Tversky loss
        # ---------------------------------------------------------------
        score = self.tvloss(rank, input, target) # In fact (1 - Tversky), which we minimize, to maximize the Tversky index

        # Hausdorff distance
        # ---------------------------------------------------------------
        if self.lambda_haus > 0:
            # Update lambda
            if epoch_cnt == True:
                self.lambda_haus = self.lambda_haus * self.haus_mult_factor

            if self.haus_option == 0: 
                # Consider all non-background labels as a single organ
                #inp = input[:, 1, ...] + input[:, 2, ...]
                inp = torch.sum(input[:, 1:, ...], axis=1)
                tar = self.one_hot_encoder(rank, target).contiguous()
                #tar = tar[:, 1, ...] + tar[:, 2, ...]
                tar = torch.sum(tar[:, 1:, ...], axis=1)
                score_haus = self.crsEnt(inp.unsqueeze(1), tar.unsqueeze(1))

                # Update the score
                score += self.lambda_haus * score_haus
            elif self.haus_option == 1: # Tested only for GPi/e!
                # Compute the Hausdorff distance per each label and side independently
                dims = input.size()
                mid = int(dims[2] / 2)

                # Left + right, for all labels (non-background)
                tar = self.one_hot_encoder(rank, target).contiguous()
                for qq in range(dims[1] - 1):
                    inp1 = input[:, qq + 1, mid:, ...]
                    tar1 = tar[:, qq + 1, mid:, ...]
                    tmp_score1 = self.crsEnt(inp1.unsqueeze(1), tar1.unsqueeze(1))

                    inp2 = input[:, qq + 1, :-mid, :, :]
                    tar2 = tar[:, qq + 1, :-mid, ...]
                    tmp_score2 = self.crsEnt(inp2.unsqueeze(1), tar2.unsqueeze(1))

                    # Update the score
                    score += self.lambda_haus * (tmp_score1 + tmp_score2)
            elif self.haus_option == 2:
                # Hausdorff distance per each organ. No division to left and right
                # Compute the Hausdorff distance per each label independently
                dims = input.size()

                # For all labels (non-background)
                tar = self.one_hot_encoder(rank, target).contiguous()
                for qq in range(dims[1] - 1):
                    inp = input[:, qq + 1, ...]
                    tar_curr = tar[:, qq + 1, ...]
                    tmp_score = self.crsEnt(inp.unsqueeze(1), tar_curr.unsqueeze(1))

                    # Update the score
                    score += self.lambda_haus * tmp_score

        # Calculate the DICE score between each input layer
        # ---------------------------------------------------------------
        smooth = 0.00001  # 0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        # input = input.view(batch_size, self.n_classes, -1) # Shouldn't this be used instead?
        target = self.one_hot_encoder(rank, target).contiguous().view(batch_size, self.n_classes, -1)

        # Calculate the DICE score between each prediction to each other label (we want to minimize overlap)
        for ii in range(self.n_classes - 1):
            input = input[:, 1:, :]
            target = target[:, 0:-1, :]

            inter = torch.sum(input * target, 2) + smooth
            # union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

            # This is the proper smoothing, as we multiply inter by 2 in the next line. This way, the loss is between [0, 1]
            union = torch.sum(input, 2) + torch.sum(target, 2) + 2.0 * smooth

            # Accumulate the score (can be extended to lambda_i)
            score += self.lam * torch.sum(2.0 * inter / union)

        return score

class registration_loss(nn.Module):
    def __init__(self):
        super(registration_loss, self).__init__()
        self.ncc_loss = ncc_loss

    def forward(self, I, J):
        # Arrange in appropriate order for ncc loss and send to loss
        loss = self.ncc_loss(I.permute(0, 2, 3, 4, 1), J.permute(0, 2, 3, 4, 1))
        return loss


class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth

    def forward(self, rank, X_in):
        self.ones = torch.sparse.torch.eye(self.depth).to(rank)

        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

if __name__ == '__main__':
    from torch.autograd import Variable
    depth=4 # 4
    batch_size=2
    encoder = One_Hot(depth=depth).forward
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    y_onehot = encoder(y)
    x = Variable(torch.randn(y_onehot.size()).float()).cuda()

    dicemetric = SoftDiceLoss(n_classes=depth)
    tvloss = TverskyLoss(n_classes=depth, alpha=0.5, beta=0.5)
    jloss = jointloss(n_classes=depth, alpha=0.5, beta=0.5, lam=1)

    q = jloss(x, y)

    # NOTE: Differneces between the DICE score and the Tversky score with [0.5, 0.5] are due to the smoothing factor
    # The lower the smoothing factor ("smooth"), the more similar DICE and Tversky will be (decimal point accuracy)
    print("DICE = {}".format(dicemetric(x,y)))
    print("Tversky = {} [alpha = {}, beta = {}]".format(tvloss(x,y), tvloss.alpha, tvloss.beta))
    print("Joint loss = {} [alpha = {}, beta = {}, lambda = {}]".format(jloss(x, y), jloss.alpha, jloss.beta, jloss.lam))

    a = 1
