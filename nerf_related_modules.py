import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


act_dict = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU, 'tanh': nn.Tanh, 'sine': Sine}


# positional Encoding
class Embedder:
    """
    kwargs include:
        include_input:  if True, the raw input is included in the embedded
        input_dim:      dimension of input to be embedded
        num_freq:       number of frequency bands
        log_sampling:   if True, the freq is linearly sampled in the log space
        max_freq_log2:  the log2 of the max freq
        periodic func:  the periodic functions used to embed input  

    """
    def __init__(self, input_dim, num_freq, max_freq_log2, include_input=True, periodic_fns=[torch.sin, torch.cos], log_sampling=True):
        super().__init__()
        self.input_dim = input_dim
        self.periodic_fns = periodic_fns
        self.include_input = include_input

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim
        self.output_dim += self.input_dim * num_freq * len(self.periodic_fns)

        if log_sampling:
            self.freq_band = 2.0 ** torch.linspace(0, max_freq_log2, num_freq)
        else:
            self.freq_band = torch.linspace(1, 2**max_freq_log2, num_freq)

    def embed(self, x):
        assert x.shape[-1] == self.input_dim
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_band:
            for fn in self.periodic_fns:
                out.append(fn(x*freq))
        out = torch.cat(out, -1)
        return out
      

class Style_NeRF_MLP(nn.Module):
    def __init__(self, W=256, D=8, input_ch_pts=3, input_ch_viewdir=3, skips=[4],
                     act_fn=nn.ReLU, use_viewdir=True, sigma_mul=0, enable_style=False):
        super().__init__()
        self.input_ch_pts = input_ch_pts
        self.input_ch_viewdir = input_ch_viewdir
        self.skips = skips
        self.act_fn = act_fn()
        self.use_viewdir = use_viewdir
        self.enable_style = enable_style

        # base layer: for density generation
        self.base_layers = nn.ModuleList()
        cur_dim = self.input_ch_pts
        for i in range(D):
            self.base_layers.append(nn.Linear(cur_dim, W))
            cur_dim = W
            if i in skips and i != D-1:
                cur_dim = W + input_ch_pts
        

        # sigma layer: generate the density
        self.sigma_layer = nn.Linear(cur_dim, 1)
        self.sigma_mul = sigma_mul

        # remap layer: for later use of rgb generation
        self.remap_layer = nn.Linear(cur_dim, 256)

        # rgb layer: generate rgb
        self.rgb_layers = nn.ModuleList()
        cur_dim = 256 + input_ch_viewdir if use_viewdir else 256
        self.rgb_layers.append(nn.Linear(cur_dim, W//2))
        self.rgb_layers.append(nn.Linear(W//2, 3))

        self.all_layers = [*self.base_layers, self.sigma_layer, self.remap_layer, *self.rgb_layers]

    def forward(self, pts, viewdirs):
        # for density
        base = self.base_layers[0](pts)
        for i in range(1, len(self.base_layers)):
            if i in self.skips:
                base = torch.cat((pts, base), dim=1)
            base = self.act_fn(self.base_layers[i](base))

        sigma = self.sigma_layer(base)
        sigma = sigma + F.relu(sigma) * self.sigma_mul

        # for rgb
        remap = F.relu(self.remap_layer(base))
        if self.use_viewdir:
            rgb = self.act_fn(self.rgb_layers[0](torch.cat((remap, viewdirs), dim=-1)))
        else:
            rgb = self.act_fn(self.rgb_layers[0](remap))
        rgb = F.sigmoid(self.remap_layer[1](rgb))

        if self.enable_style:
            out= OrderedDict([( 'rgb', rgb),  ('pts', pts), ('sigma', sigma.squeeze(-1))])
        else:
            out= OrderedDict([( 'rgb', rgb),  ('sigma', sigma.squeeze(-1))])
        return out

            

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
    