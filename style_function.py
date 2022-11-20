import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaIN(feat_content, feat_style):
    assert(feat_content.size()[:2] == feat_style.szie()[:2])
