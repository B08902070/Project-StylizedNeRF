import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        # super(InfiniteSamplerWrapper, self).__init__()
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31




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
        rand.uniform_(0, 1)
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
    t_vals, _ = torch.sort(torch.cat([ts, t_samples], -1), -1)
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
    if torch.cuda.is_available():
        denom = denom.cuda()
    t = (u-cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


