# dataset.py is for preparing and generating dataset for training style nerf
import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from nst_net import NST_Net



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


def style_data_prepare(style_path, content_images, size=512, chunk=64, sv_path=None, decoder_dir='./pretrained/decoder/', save_geo=True, no_reload=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """nst net"""
    encoder_path = './pretrained/vgg_normalised.pth'
    nst_net = NST_Net(encoder_pretrained_path=encoder_path)
    ckpts = [os.path.join(decoder_dir, f) for f in sorted(os.listdir(decoder_dir)) if 'decoder_iter_' in f]
    if len(ckpts) > 0 and not no_reload:
        print(f'loading {ckpts[-1]}')
        ld_dict = torch.load(ckpts[-1])
        nst_net.load_decoder_state_dict(ld_dict['decoder'])
    else:
        print('Please finetune decoder first')
        exit(0)
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
                _, _, tmp_stylized_imgs, tmp_style_features = nst_net(content=tmp_imgs, style=style_img[:tmp_imgs.shape[0]], alpha=1, return_img_and_feat = True)
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
