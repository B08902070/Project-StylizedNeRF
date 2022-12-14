import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        out = torch.cat(out, -1).type(torch.FloatTensor)
        return out
      

class Style_NeRF_MLP(nn.Module):
    def __init__(self, W=256, D=8, input_ch_pts=3, input_ch_viewdir=3, skips=[4], \
                     act_fn=nn.ReLU, use_viewdir=True, sigma_mul=0):
        super().__init__()
        self.input_ch_pts = input_ch_pts
        self.input_ch_viewdir = input_ch_viewdir
        self.skips = skips
        self.act_fn = act_fn()
        self.use_viewdir = use_viewdir

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
            base = self.act_fn(self.base_layers[i](base))
            if i in self.skips:
                base = torch.cat((pts, base), dim=-1)
            

        sigma = self.sigma_layer(base)
        sigma = sigma + F.relu(sigma) * self.sigma_mul

        # for rgb
        remap = F.relu(self.remap_layer(base))
        if self.use_viewdir:
            rgb = self.act_fn(self.rgb_layers[0](torch.cat((remap, viewdirs), dim=-1)))
        else:
            rgb = self.act_fn(self.rgb_layers[0](remap))
        rgb = F.sigmoid(self.rgb_layers[1](rgb))

        return { 'rgb': rgb,  'sigma': sigma.squeeze(-1)}


def sampling_pts_uniform(rays_o, rays_d, N_samples=64, near=0., far=1.05, harmony=False, perturb=False):
    #  Intersect, ts_nf of shape [ray, box] and [ray, box, 2]
    ray_num = rays_o.shape[0]

    #  Uniform sampling ts of shape [ray, N_samples]
    ts = torch.linspace(0, 1, N_samples).unsqueeze(0).expand(ray_num, N_samples)
    if not harmony:
        ts = ts * (far - near) + near
    else:
        ts = 1. / (1./near * (1 - ts) + 1./far * ts)

    if perturb:
        #  Add perturb
        rand = torch.zeros([ray_num, N_samples])
        nn.init.uniform_(rand, 0, 1)
        mid = (ts[..., 1:] + ts[..., :-1]) / 2
        upper = torch.cat([mid, ts[..., -1:]], -1)
        lower = torch.cat([ts[..., :1], mid], -1)
        ts = lower + (upper - lower) * rand

    #  From ts to pts. [ray, N_samples, 3]
    rays_o, rays_d = rays_o.unsqueeze(1).expand([ray_num, N_samples, 3]), rays_d.unsqueeze(1).expand([ray_num, N_samples, 3])
    ts_expand = ts.unsqueeze(-1).expand([ray_num, N_samples, 3])
    pts = rays_o + ts_expand * rays_d

    return pts, ts



def sampling_pts_fine(rays_o, rays_d, ts, weights, N_samples_fine=64):

    # ts of shape [ray, N_samples], ts_mid of shape [ray, N_samples - 1]
    ts_mid = 0.5 * (ts[..., 1:] + ts[..., :-1])
    t_samples = sample_pdf(ts_mid, weights[..., 1:-1], N_samples_fine, det=True)
    t_samples = t_samples.detach()
    _, t_vals = torch.argsort(torch.cat([ts, t_samples], -1), -1)
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)  # [N_rays, N_samples + N_importance, 3]

    # Avoid BP
    t_vals = t_vals.detach()

    return pts, t_vals


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdims=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.random(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.maximum(torch.zeros_like(inds-1), inds-1)
    above = torch.minimum((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0]).cpu()
    cond = np.where(denom < 1e-5)
    denom[cond[0], cond[1]] = 1.
    denom = denom.cuda()
    t = (u-cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def batchify(fn, chunk=1024*32):
    """Render rays in smaller minibatches to avoid OOM.
    """
    if chunk is None:
        return fn

    def ret_func(**kwargs):
        x = kwargs[list(kwargs.keys())[0]]
        all_ret = {}
        for i in range(0, x.shape[0], chunk):
            end = min(i + chunk, x.shape[0])
            chunk_kwargs = dict([[key, kwargs[key][i: end]] for key in kwargs.keys()])
            ret = fn(**chunk_kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    return ret_func



def alpha_composition(pts_rgb, pts_sigma, t_values, sigma_noise_std=0., white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        pts_rgb: [num_rays, num_samples along ray, 3]. Prediction from model.
        pts_sigma: [num_rays, num_samples along ray]. Prediction from model.
        t_values: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_exp: [num_rays, 3]. Estimated RGB color of a ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        t_exp: [num_rays]. Estimated distance to object.
    """
    sigma2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    delta = t_values[..., 1:] - t_values[..., :-1]
    delta = torch.cat([delta, torch.Tensor([1e10]).expand(delta[..., :1].shape)], -1)  # [N_rays, N_samples]

    noise = 0.
    if sigma_noise_std > 0:
        noise = torch.randn(size = pts_sigma.shape, dtype = pts_sigma.dtype) * sigma_noise_std

    alpha = sigma2alpha(F.relu(pts_sigma + noise), delta)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_exp = torch.sum(weights[..., None] * pts_rgb, -2)  # [N_rays, 3]

    t_exp = torch.sum(weights * t_values, -1)
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_exp = rgb_exp + (1. - acc_map[..., None])

    return rgb_exp, t_exp, weights