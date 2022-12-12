import torch
import torch.nn as nn
import torch.nn.functional as F

from nerf_helper import Embedder, Style_NeRF_MLP



act_dict = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU, 'tanh': nn.Tanh}
          

class Style_NeRF(nn.Module):
    def __init__(self, args, mode='coarse', enable_style=False):
        self.use_viewdir = args.use_viewdir
        self.act_fn = act_dict[args.act_type]

        # embedding
        self.embedder_coor = Embedder(input_dim=3, num_freq=args.embed_freq_coor, max_freq_log2=args.embed_freq_coor-1)
        self.embedder_dir = Embedder(input_dim=3, num_freq=args.embed_freq_dir, max_freq_log2=args.embed_freq_dir-1)
        self.input_ch_pts = self.embedder_coor.output_dim
        self.input_ch_viewdir = self.embedder_dir.output_dim
        self.skips=[4]
        self.sigma_mul=0


        # Neural Net
        if mode == 'coarse':
            self.netdepth = args.netdepth
            self.netwidth = args.netwidth
        else:
            self.netdepth = args.netdepth_fine
            self.netwidth = args.netwidth_fine

        self.mlp = Style_NeRF_MLP(W=self.netwidth, D=self.netdepth, input_ch_pts=self.input_ch_pts, input_ch_viewdir=self.input_ch_viewdir, \
                             skips=self.skips, act_fn=self.act_fn, use_viewdir=self.use_viewdir, sigma_mul=self.sigma_mul)        


    def forward(self, pts, dirs):
        emb_pts = self.embedder_coor(pts)
        emb_dirs = self.embedder_dir(dirs)

        out = self.mlp(emb_pts, emb_dirs)
        out['dirs'] = dirs
        out['pts'] = emb_pts
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
    