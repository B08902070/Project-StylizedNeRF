import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_mean_std(feat, eps=1e-5):
    assert(len(feat.size()) == 4)
    N, C = feat.size()[:2]
    var_feat = feat.var(dim=[2, 3]) + eps
    std_feat = var_feat.sqrt().view(N, C, 1, 1)
    mean_feat = feat.mean(dim=[2, 3]).view(N, C, 1, 1)

    return mean_feat, std_feat

def adaIN(feat_content, feat_style):
    assert(feat_content.size()[:2] == feat_style.size()[:2])
    size = feat_content.size()
    mean_content, std_content = cal_mean_std(feat_content)
    mean_style, std_style = cal_mean_std(feat_style)

    feat_stylized = (feat_content-mean_content.expand(size))/std_content.expand(size)
    feat_stylized = feat_stylized * std_style.expand(size) + mean_style.expand(size)

    return feat_stylized

def content_loss(feat_input, feat_target):
    assert (feat_input.size() == feat_target.size())
    mse_loss = nn.MSELoss()
    return mse_loss(feat_input, feat_target)

def style_loss(feats_input, feats_target):
    assert (len(feats_input) == len(feats_target))
    loss = 0.0
    mse_loss = nn.MSELoss()
    for i in range(len(feats_input)):
        assert(feats_input[i].size() == feats_target[i].size())
        mean_input, std_input = cal_mean_std(feats_input[i])
        mean_target, std_target = cal_mean_std(feats_target[i])
        loss += (mse_loss(mean_input, mean_target) + mse_loss(std_input, std_target))
    return loss
