# dataset.py is for preparing and generating dataset for training style nerf
import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

import VGG
from nst_net import NST_Net
from load_llff import load_llff_data



def view_synthesis(cps, factor=10):
    frame_num = cps.shape[0]
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R
    from scipy import interpolate as intp
    rots = R.from_matrix(cps[:, :3, :3])
    slerp = Slerp(np.arange(frame_num), rots)
    tran = cps[:, :3, -1]
    f_tran = intp.interp1d(np.arange(frame_num), tran.T)

    new_num = int(frame_num * factor)

    new_rots = slerp(np.linspace(0, frame_num - 1, new_num)).as_matrix()
    new_trans = f_tran(np.linspace(0, frame_num - 1, new_num)).T

    new_cps = np.zeros([new_num, 4, 4], np.float)
    new_cps[:, :3, :3] = new_rots
    new_cps[:, :3, -1] = new_trans
    new_cps[:, 3, 3] = 1
    return new_cps


# generate rays that pass through the center of each pixel of the images
def get_rays(H, W, K, c2w, pixel_alignment=True):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i, j = i.t(), j.t()

    if pixel_alignment:
        i, j = i+0.5, j+0.5
    
    # image coor to camera coor
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)

    # get the direction of the ray
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # get the origin of the ray
    rays_o = c2w.expand(rays_d.shape)

    return rays_o, rays_d

def get_rays_np(H, W, K, c2w, pixel_alignment=True):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    if pixel_alignment:
        i, j = i+0.5, j+0.5
    
    # image coor to camera coor
    dirs = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], -np.ones_like(i)], -1)

    # get the direction of the ray
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # get the origin of the ray
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)

    return rays_o, rays_d
    
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1/(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1/(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1 + 2 * near / rays_o[..., 2]

    d0 = -1/(W/(2*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1/(H/(2*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

def ndc_rays_np(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1/(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1/(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1 + 2 * near / rays_o[..., 2]

    d0 = -1/(W/(2*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1/(H/(2*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2 * near / rays_o[..., 2]

    rays_o = np.stack([o0, o1, o2], -1)
    rays_d = np.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def image_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_data_prepare(style_path, content_images, size=512, chunk=64, sv_path=None, decode_path='./pretrained/decoder.pth', save_geo=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """nst net"""
    encoder_path = './pretrained/vgg_normalised.pth'
    nst_net = NST_Net(encoder_pretrained_path=encoder_path)
    decoder_data = torch.load(decode_path)
    if 'decoder' in decoder_data.keys():
        nst_net.load_decoder_state_dict(decoder_data['decoder'])
    else:
        nst_net.load_decoder_state_dict(decoder_data)
    nst_net.eval()
    nst_net.to(device)

    images_path = glob.glob(style_path + '/*.png') + glob.glob(style_path + '/*.jpg') + glob.glob(style_path + '/*.jpeg') + glob.glob(style_path + '/*.JPG') + glob.glob(style_path + '/*.PNG')
    print(style_path, images_path)
    style_images, style_paths, style_names = [], [], {}
    style_features = np.zeros([len(images_path), 1024], dtype=np.float32)
    img_trans = image_transform(size)
    for i in tqdm(range(len(images_path))):
        images_path[i] = images_path[i].replace('\\', '/')
        print("Style Image: " + images_path[i])

        """Read Style Images"""
        style_img = img_trans(Image.open(images_path[i]))  # become tensor
        style_images.append(np.moveaxis(style_img.numpy(), 0, -1))

        """Stylization"""
        stylized_images = np.zeros_like(content_images)
        style_feature = np.zeros([1024], dtype=np.float32)
        style_img = style_img.float().to(device).unsqueeze(0).expand([chunk, *style_img.shape])
        start = 0
        while start < content_images.shape[0]:
            end = min(start + chunk, content_images.shape[0])
            tmp_imgs = torch.movedim(torch.from_numpy(content_images[start: end]).float().to(device), -1, 1)
            with torch.no_grad():
                tmp_stylized_imgs, tmp_style_features = nst_net(content=tmp_imgs, style=style_img[:tmp_imgs.shape[0]], alpha=1, return_img_and_feat = True)
                tmp_stylized_imgs = np.moveaxis(tmp_stylized_imgs.cpu().numpy(), 1, -1)
            for j in range(end-start):
                stylized_images[start+j] = cv2.resize(tmp_stylized_imgs[j], (stylized_images.shape[2], stylized_images.shape[1]))
            style_feature = np.concatenate([tmp_style_features[0].reshape(-1, 512).mean(dim=0).cpu().numpy(), tmp_style_features[0].reshape([-1, 512]).var(dim=0).cpu().numpy()])
            start = end

        """Stylized Images Saving"""
        style_name = images_path[i].split('/')[-1].split('.')[0]
        style_names[style_name] = i
        if sv_path is not None:
            if not os.path.exists(sv_path + '/' + style_name):
                os.makedirs(sv_path + '/' + style_name)
            for j in range(stylized_images.shape[0]):
                Image.fromarray(np.array(stylized_images[j] * 255, np.uint8)).save(sv_path + '/' + style_name + '/%03d.png' % j)
                if save_geo:
                    np.savez(sv_path + '/' + style_name + '/%03d' % j, stylized_image=stylized_images[j])
        style_paths.append(sv_path + '/' + style_name)
        style_features[i] = style_feature
    style_images = np.stack(style_images)

    return style_names, style_paths, style_images, style_features


class RaySampler(Dataset):
    def __init__(self, data_path, factor=2., mode='train', valid_factor=3, no_ndc=False, pixel_alignment=False, spherify=False):
        super().__init__()

        images, poses, bds, render_poses, i_test = load_llff_data(data_path, factor, recenter=True, bd_factor=.75, spherify=spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, data_path)
        print('DEFINING BOUNDS')
        if no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

        """Validation Rays"""
        cps = np.concatenate([poses[:, :3, :4], np.zeros_like(poses[:, :1, :])], axis=1)
        cps[:, 3, 3] = 1.
        

        print('K:', K)
        print('Camera Pose: ', cps.shape)

        """Setting Attributes"""
        self.mode = mode
        self.frame_num = cps.shape[0]
        self.h, self.w, self.f = H, W, focal
        self.hwf = hwf
        self.K = K
        self.cx, self.cy = W / 2., H / 2.
        self.cps, self.intr, self.images = cps, K, images
        self.cps_valid = view_synthesis(cps, valid_factor) 
        self.rays_num = self.frame_num * self.h * self.w
        self.near, self.far = near, far
        self.rays_o, self.rays_d = None, None
        self.pixel_alignment = pixel_alignment
        self.no_ndc = no_ndc

        self._gen_rays(mode)

    def _gen_rays(self, mode):
        if mode == 'train':
            print('get rays of training')
            if self.rays_o != None and self.rays_d != None:
                del(self.rays_o);   del(self.rays_d)
            self.rays_o, self.rays_d = np.zeros([self.cps.shape[0], self.h, self.w, 3]), np.zeros([self.cps.shape[0], self.h, self.w, 3])
            for i in tqdm(range(self.cps.shape[0])):
                tmp_rays_o, tmp_rays_d = get_rays_np(self.h, self.w, self.K, self.cps[i, :3, :4], self.pixel_alignment)
                self.rays_o[i] = tmp_rays_o
                self.rays_d[i] = tmp_rays_d
        else:
            print('get rays of validation')
            if self.rays_o != None and self.rays_d != None:
                del(self.rays_o);   del(self.rays_d)
            self.rays_o, self.rays_d = np.zeros([self.cps_valid.shape[0], self.h, self.w, 3]), np.zeros([self.cps_valid.shape[0], self.h, self.w, 3])
            for i in tqdm(range(self.cps_valid.shape[0])):
                tmp_rays_o, tmp_rays_d = get_rays_np(self.h, self.w, self.K, self.cps_valid[i, :3, :4], self.pixel_alignment)
                self.rays_o[i] = tmp_rays_o
                self.rays_d[i] = tmp_rays_d
        if not self.no_ndc:
            self.rays_o, self.rays_d = ndc_rays_np(self.h, self.w, self.K[0][0], 1., self.rays_o, self.rays_d)


    def _my_get_item(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        rgb = self.images[frame_id, hid, wid]
        ray_o = self.rays_o[frame_id, hid, wid]
        ray_d = self.rays_d[frame_id, hid, wid]
        if self.mode == 'train':
            return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d}
        else:
            return {'rays_o': ray_o, 'rays_d': ray_d}


    def set_mode(self, mode='train'):
        modes = ['train', 'valid']
        if mode not in modes:
            print('Unknown mode: ', mode, ' Only supports: ', modes)
            exit(-1)
        if mode != self.mode:
            self._gen_rays(mode)

        self.mode = mode

    def __getitem__(self, item):
        return self._my_get_item(item)

    def __len__(self):
        if self.mode == 'train':
            return self.frame_num * self.w * self.h
        else:
            return self.cps_valid.shape[0] * self.w * self.h


class StyleRaySampler(Dataset):
    def __init__(self, data_path, style_path, factor=2., mode='train', valid_factor=3, dataset_type='llff', no_ndc=False, pixel_alignment=False, spherify=False, TT_far=4.):
        super().__init__()

        images, poses, bds, render_poses, i_test = load_llff_data(data_path, factor, recenter=True, bd_factor=.75, spherify=spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, data_path)
        print('DEFINING BOUNDS')
        if no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        K = np.array([
               [focal, 0, 0.5*W],
               [0, focal, 0.5*H],
               [0, 0, 1]
           ])

        """Validation Rays"""
        cps = np.concatenate([poses[:, :3, :4], np.zeros_like(poses[:, :1, :])], axis=1)
        cps[:, 3, 3] = 1.

        """Style Data"""
        if not os.path.exists(data_path + '/stylized_' + str(factor) + '/' + '/stylized_data.npz'):
            print("Stylizing training data ...")
            style_names, style_paths, style_images, style_features = style_data_prepare(style_path, images, size=512, chunk=8, sv_path=data_path + '/stylized_' + str(factor) + '/', decode_path='./pretrained/decoder.pth')
            np.savez(data_path + '/stylized_' + str(factor) + '/' + '/stylized_data', style_names=style_names, style_paths=style_paths, style_images=style_images, style_features=style_features)
        else:
            print("Stylized data from " + data_path + '/stylized_' + str(factor) + '/' + '/stylized_data.npz')
            stylized_data = np.load(data_path + '/stylized_' + str(factor) + '/' + '/stylized_data.npz', allow_pickle=True)
            style_names, style_paths, style_images, style_features = stylized_data['style_names'], stylized_data['style_paths'], stylized_data['style_images'], stylized_data['style_features']
            print("Dataset Creation Done !")

        """Setting Attributes"""
        self.mode = mode
        self.frame_num = cps.shape[0]
        self.h, self.w, self.f = H, W, focal
        self.hwf = hwf
        self.K = K
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.cps, self.intr, self.images = cps, K, images
        self.cps_valid = view_synthesis(cps, valid_factor) 
        self.rays_num = self.frame_num * self.h * self.w
        self.near, self.far = near, far

        self.style_names = style_names
        self.style_images = style_images
        self.style_features = style_features
        self.style_paths = style_paths

        self.style_num = self.style_images.shape[0]
        self.rays_o, self.rays_d = None, None
        self.pixel_alignment = pixel_alignment
        self.no_ndc = no_ndc

        self._gen_rays(mode)

    def _gen_rays(self, mode):      
        if mode == 'train':
            print('get rays of training')
            if self.rays_o == None and self.rays_d == None:
                self.rays_o, self.rays_d = np.zeros([self.cps.shape[0], self.h, self.w, 3]), np.zeros([self.cps.shape[0], self.h, self.w, 3])
            for i in tqdm(range(self.cps.shape[0])):
                tmp_rays_o, tmp_rays_d = get_rays_np(self.h, self.w, self.K, self.cps[i, :3, :4], self.pixel_alignment)
                self.rays_o[i] = tmp_rays_o
                self.rays_d[i] = tmp_rays_d
        else:
            print('get rays of validation')
            if self.rays_o == None and self.rays_d == None:   
                self.rays_o, self.rays_d = np.zeros([self.cps_valid.shape[0], self.h, self.w, 3]), np.zeros([self.cps_valid.shape[0], self.h, self.w, 3])
            for i in tqdm(range(self.cps_valid.shape[0])):
                tmp_rays_o, tmp_rays_d = get_rays_np(self.h, self.w, self.K, self.cps_valid[i, :3, :4], self.pixel_alignment)
                self.rays_o[i] = tmp_rays_o
                self.rays_d[i] = tmp_rays_d
        if not self.no_ndc:
            self.rays_o, self.rays_d = ndc_rays_np(self.h, self.w, self.K[0][0], 1., self.rays_o, self.rays_d)

    def _my_get_item(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        rgb = self.images[frame_id, hid, wid]
        ray_o = self.rays_o[frame_id, hid, wid]
        ray_d = self.rays_d[frame_id, hid, wid]
        if self.mode == 'train':
            return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d}
        else:
            return {'rays_o': ray_o, 'rays_d': ray_d}

    def _my_get_item_style(self, idx):
        style_id = idx // (self.frame_num * self.h * self.w)
        frame_id = (idx % (self.frame_num * self.h * self.w)) // (self.h * self.w)
        hid = (idx % (self.h * self.w)) // self.w
        wid = idx % self.w
        stylized_contents = np.load(self.style_paths[style_id] + '/%03d.npz' % frame_id)['stylized_image']
        rgb = stylized_contents[hid, wid]
        rgb_origin = self.images[frame_id, hid, wid]
        style_feature = self.style_features[style_id]
        ray_o = self.rays_o[frame_id, hid, wid]
        ray_d = self.rays_d[frame_id, hid, wid]
        style_image = torch.from_numpy(self.style_images[style_id]).float()
        if self.mode == 'train_style':
            return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d, 'style_feature': style_feature, 'rgb_origin': rgb_origin, 'style_id': style_id, 'frame_id': frame_id}
        else:
            return {'rays_o': ray_o, 'rays_d': ray_d, 'style_image': style_image, 'style_id': style_id, 'frame_id': frame_id}


    def set_mode(self, mode='train'):
        modes = ['train', 'valid', 'train_style', 'valid_style']
        if mode not in modes:
            print('Unknown mode: ', mode, ' Only supports: ', modes)
            exit(-1)
        if mode != self.mode:
            self._gen_rays(mode)
        self.mode = mode

    def __getitem__(self, item):
        if self.mode in ['train', 'valid']:
            return self._my_get_item(item)
        else:
            return self._my_get_item_style(item)

    def __len__(self):
        if self.mode == 'train':
            return self.frame_num * self.w * self.h
        elif self.mode == 'valid':
            return self.cps_valid.shape[0] * self.w * self.h
        elif self.mode == 'train_style':
            return self.style_num * self.frame_num * self.w * self.h
        else:
            return self.style_num * self.cps_valid.shape[0] * self.w * self.h



class StyleRaySampler_gen(Dataset):
    def __init__(self, data_path, style_path, gen_path, factor=2., mode='train', valid_factor=0.05, dataset_type='llff', no_ndc=False, pixel_alignment=False, spherify=False, decode_path='./pretrained/decoder.pth', TT_far=4., collect_stylized_images=True):
        super().__init__()

        K = None
        if dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(data_path, factor, recenter=True, bd_factor=.75, spherify=spherify)
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, data_path)
            print('DEFINING BOUNDS')
            if no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        else:
            poses = hwf = K = near = far = None
            print('Unknown dataset type', dataset_type, 'exiting')
            exit(0)

        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        self.gen_path = gen_path
        self.image_paths = sorted(list(Path(self.gen_path).glob('rgb_*.png')))
        self.geo_paths = sorted(list(Path(self.gen_path).glob('geometry_*.npz')))
        data = np.load(str(self.geo_paths[0]))
        self.hwf = data['hwf']
        frame_num = len(self.image_paths)
        images = np.zeros([frame_num, H, W, 3], np.float32)
        cps = np.zeros([frame_num, 4, 4], np.float32)
        for i in range(frame_num):
            images[i] = np.array(Image.open(str(self.image_paths[i])).convert('RGB'), dtype=np.float32) / 255.
            cps[i] = np.load(str(self.geo_paths[i]))['cps']

        """Validation Rays"""
        cps_valid = view_synthesis(cps, valid_factor)

        rays_o, rays_d = np.zeros([cps.shape[0], H, W, 3]), np.zeros([cps.shape[0], H, W, 3])
        for i in tqdm(range(cps.shape[0])):
            tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps[i, :3, :4], pixel_alignment)
            rays_o[i] = tmp_rays_o
            rays_d[i] = tmp_rays_d
        rays_o_valid, rays_d_valid = np.zeros([cps_valid.shape[0], H, W, 3]), np.zeros([cps_valid.shape[0], H, W, 3])
        for i in tqdm(range(cps_valid.shape[0])):
            tmp_rays_o, tmp_rays_d = get_rays_np(H, W, K, cps_valid[i, :3, :4], pixel_alignment)
            rays_o_valid[i] = tmp_rays_o
            rays_d_valid[i] = tmp_rays_d

        if dataset_type == 'llff' and not no_ndc:
            rays_o, rays_d = ndc_rays_np(H, W, K[0][0], 1., rays_o, rays_d)
            rays_o_valid, rays_d_valid = ndc_rays_np(H, W, K[0][0], 1., rays_o_valid, rays_d_valid)

        """Style Data"""
        if not os.path.exists(data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data.npz'):
            print("Stylizing training data ...")
            style_names, style_paths, style_images, style_features = style_data_prepare(style_path, images, size=512, chunk=8, sv_path=data_path + '/stylized_gen_' + str(factor) + '/', decode_path=decode_path)
            np.savez(data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data', style_names=style_names, style_paths=style_paths, style_images=style_images, style_features=style_features)
        else:
            print("Stylized data from " + data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data.npz')
            stylized_data = np.load(data_path + '/stylized_gen_' + str(factor) + '/' + '/stylized_data.npz', allow_pickle=True)
            style_names, style_paths, style_images, style_features = stylized_data['style_names'][()], stylized_data['style_paths'], stylized_data['style_images'], stylized_data['style_features']

        """Setting Attributes"""
        self.set_mode(mode)
        self.frame_num = cps.shape[0]
        self.h, self.w, self.f = H, W, focal
        self.hwf = hwf
        self.K = K
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.cps, self.intr, self.images = cps, K, images
        self.cps_valid = cps_valid
        self.rays_num = self.frame_num * self.h * self.w
        self.near, self.far = near, far

        self.style_names = style_names
        self.style_names_t = {y: x for x, y in self.style_names.items()}
        self.style_images = style_images
        self.style_paths = style_paths
        self.style_features = style_features
        self.style_num = self.style_images.shape[0]

        self.is_ndc = (dataset_type == 'llff' and not no_ndc)
        self.rays_o_dict = {'train': rays_o, 'valid':rays_o_valid, 'train_style':rays_o, 'valid_style':rays_o_valid}
        self.rays_d_dict = {'train': rays_d, 'valid':rays_d_valid, 'train_style':rays_d, 'valid_style':rays_d_valid}
        self.stylized_images_uint8 = None
        if collect_stylized_images:
            self.collect_all_stylized_images()
        print("Dataset Creation Done !")

    def collect_all_stylized_images(self):
        print(self.style_names.keys())
        if self.stylized_images_uint8 is not None:
            return
        self.stylized_images_uint8 = np.zeros([self.style_num, self.frame_num, self.h, self.w, 3], dtype=np.uint8)
        for i in range(self.style_num):
            print('Collecting style: ' + self.style_names_t[i])
            for j in tqdm(range(self.frame_num)):
                img = np.array(Image.open(self.style_paths[i] + '/%03d.png' % j).convert('RGB'), np.uint8)
                self.stylized_images_uint8[i, j] = img

    def _my_get_item(self, idx):
        frame_id = idx // (self.h * self.w)
        pixel_id = idx % (self.h * self.w)
        hid, wid = pixel_id // self.w, pixel_id % self.w
        rgb = self.images[frame_id, hid, wid]
        ray_o = self.rays_o_dict[self.mode][frame_id, hid, wid]
        ray_d = self.rays_d_dict[self.mode][frame_id, hid, wid]

        if self.mode == 'train':
            return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d}
        else:
            return {'rays_o': ray_o, 'rays_d': ray_d}

    def _my_get_item_style(self, idx):
        style_id = idx // (self.frame_num * self.h * self.w)
        frame_id = (idx % (self.frame_num * self.h * self.w)) // (self.h * self.w)
        hid = (idx % (self.h * self.w)) // self.w
        wid = idx % self.w
        if self.stylized_images_uint8 is None:
            stylized_contents = np.load(self.style_paths[style_id] + '/%03d.npz' % frame_id)['stylized_image']
            rgb = stylized_contents[hid, wid]
        else:
            rgb = np.float32(self.stylized_images_uint8[style_id, frame_id, hid, wid]) / 255
        rgb_origin = self.images[frame_id, hid, wid]
        style_feature = self.style_features[style_id]

        ray_o = self.rays_o_dict[self.mode][frame_id, hid, wid]
        ray_d = self.rays_d_dict[self.mode][frame_id, hid, wid]

        if self.mode == 'train_style':
            return {'rgb_gt': rgb, 'rays_o': ray_o, 'rays_d': ray_d, 'style_feature': style_feature, 'rgb_origin': rgb_origin, 'style_id': style_id, 'frame_id': frame_id}
        else:
            return {'rays_o': ray_o, 'rays_d': ray_d, 'style_feature': style_feature, 'style_id': style_id, 'frame_id': frame_id}
  

    def set_mode(self, mode='train'):
        modes = ['train', 'valid', 'train_style', 'valid_style']
        if mode not in modes:
            print('Unknown mode: ', mode, ' Only supports: ', modes)
            exit(-1)
        self.mode = mode

    def __getitem__(self, item):
        if self.mode in ['train', 'valid']:
            return self._my_get_item(item)
        else:
            return self._my_get_item_style(item)

    def __len__(self):
        if self.mode == 'train':
            return self.frame_num * self.w * self.h
        elif self.mode == 'valid':
            return self.cps_valid.shape[0] * self.w * self.h
        elif self.mode == 'train_style':
            return self.style_num * self.frame_num * self.w * self.h
        else:
            return self.style_num * self.cps_valid.shape[0] * self.w * self.h


class LightDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_num = len(dataset)
        self.data_idx = np.arange(self.data_num)
        if self.shuffle:
            np.random.shuffle(self.data_idx)
        self.start = 0
        data0 = self.dataset.__getitem__(0)
        self.keys = data0.keys()

    def get_batch(self):
        if self.batch_size >= self.data_num:
            idx = np.random.choice(self.data_idx, self.batch_size, replace=True)
            # Initialize
            batch_data = {}
            for key in self.keys:
                batch_data[key] = []
            # Append data
            for i in range(self.batch_size):
                data = self.dataset.__getitem__(idx[i])
                for key in data.keys():
                    batch_data[key].append(data[key])
            self.start += self.batch_size
            # To tensor
            for key in self.keys:
                batch_data[key] = torch.from_numpy(np.stack(batch_data[key]))
            return batch_data

        # Check if shuffle again
        if self.start + self.batch_size >= self.data_num:
            self.start = 0
            np.random.shuffle(self.data_idx)
        # Initialize
        batch_data = {}
        for key in self.keys:
            batch_data[key] = []
        # Append data
        for i in range(self.batch_size):
            data = self.dataset.__getitem__(self.data_idx[self.start + i])
            for key in data.keys():
                batch_data[key].append(data[key])
        self.start += self.batch_size
        # To tensor
        for key in self.keys:
            batch_data[key] = torch.from_numpy(np.stack(batch_data[key])).float()
        return batch_data