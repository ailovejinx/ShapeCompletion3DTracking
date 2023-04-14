"""
  @ Author       : Ailovejinx
  @ Date         : 2023-04-13 09:44:59
  @ LastEditors  : Ailovejinx
  @ LastEditTime : 2023-04-14 14:16:23
  @ FilePath     : PCLosses.py
  @ Description  : 
  @ Copyright (c) 2023 by Ailovejinx, All Rights Reserved. 
"""
import torch
import torch.nn as nn


# Calculate the pairwise distance between point clouds in a batch
# 计算点云中的点两两之间的距离
def batch_pairwise_dist(x, y, use_cuda=True):
    x = x.transpose(2, 1)
    y = y.transpose(2, 1)
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    # bmm是batch矩阵乘法
    # 两个输入tensor维度是( b × n × m )和( b × m × p ), 
    # 第一维b代表batch size，输出为( b × n × p )
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
        zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


# Calculate Chamfer Loss
class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        # preds & gts of size (BS, 3, N)
        P = batch_pairwise_dist(preds, gts, self.use_cuda)
        # P of size (BS, 3, N)
        mins, _ = torch.min(P, 1)   # torch.min(input, dim, keepdim=False, *, out=None)
        # torch.min()返回最小值和索引
        loss_1 = torch.sum(mins)  # sum of all batches
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)  # sum of all batches

        return loss_1 + loss_2


# Calculate accuracy and completeness between two point clouds
def acc_comp(preds, gts, rho=0.02):
    P = batch_pairwise_dist(preds, gts).abs().sqrt()
    pred_mins, _ = torch.min(P, 2)
    gts_mins, _ = torch.min(P, 1)
    acc = pred_mins.mean(dim=1, dtype=torch.float)
    comp = gts_mins.mean(dim=1, dtype=torch.float)
    return acc, comp
