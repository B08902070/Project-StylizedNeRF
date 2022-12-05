import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from nerf_helper import Embedder, Style_NeRF_MLP

class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


act_dict = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU, 'tanh': nn.Tanh, 'sine': Sine}
          

class Style_NeRF(nn.Module):
    def __init__(self, args, mode='coarse', enable_style=False):
        self.use_viewdir = args.use_viewdir
        self.act_fn = act_dict[args.act_type]
        self.is_siren = (args.act_type == 'sine')
        self.enable_style = enable_style

        # embedding
        if not self.is_siren:
            self.embedder_coor = Embedder(input_dim=3, num_freq=args.embed_freq_coor, max_freq_log2=args.embed_freq_coor-1)
            self.embedder_dir = Embedder(input_dim=3, num_freq=args.embed_freq_dir, max_freq_log2=args.embed_freq_dir-1)
            self.input_ch_pts = self.embedder_coor.output_dim
            self.input_ch_viewdir = self.embedder_dir.output_dim
            self.skips=[4]
            self.sigma_mul=0
        else:
            self.input_ch_pts = 3
            self.input_ch_viewdir = 3
            self.skips = []
            self.sigma_mul = args.siren_sigma_mul

        # Neural Net
        if mode == 'coarse':
            self.netdepth = args.netdepth
            self.netwidth = args.netwidth
        else:
            self.netdepth = args.netdepth_fine
            self.netwidth = args.netwidth_fine
        self.net = Style_NeRF_MLP(W=self.netwidth, D=self.netdepth, input_ch_pts=self.input_ch_pts, input_ch_viewdir=self.input_ch_viewdir,
                             skips=self.skips, act_fn=self.act_fn, use_viewdir=self.use_viewdir, sigma_mul=self.sigma_mul, enable_style=enable_style)        

    def set_enable_style(self, enable_style):
        self.enable_style = enable_style
        self.net.enable_style = enable_style

    def forward(self, pts, dirs):
        if not self.is_siren:
            pts = self.embedder_coor(pts)
            dirs = self.embedder_dir(dirs)

        out = self.net(pts, dirs)
        out['dirs'] = dirs
        return out

class Style_Module(nn.Module):
    def __init__(self, args, mode='coarse'):
        super().__init__()
        self.D = args.style_D
        self.W = args.netwith
        self.input_ch = args.embed_freq_coor * 3 * 2 + 3 + args.vae_latent
        self.skips = [4]
        self.layers=nn.ModuleList()
        cur_dim = self.input_ch
        for i in range(self.D-1):
            if i in self.skips:
                dim += args.embed_freq_coor*3*2+3
            self.layers.append(nn.Linear(cur_dim, self.W))
            cur_dim = self.W
        self.layers.append(nn.Linear(cur_dim, 3))

    def forward(self, **kwargs):
        x = kwargs['x']
        l = kwargs['latent']
        h = x
        for i in range(len(self.layers)-1):
            h = torch.cat((h, l), dim=-1)
            if i in self.skips:
                h = torch.cat((h, x), dim=-1)
            h = F.relu(self.layers[i](h))
        h = torch.cat((h, l), dim=-1)
        h = self.layers[-1](h)
        h = F.sigmoid(h)
        out = {'rgb': h}
        return out
    