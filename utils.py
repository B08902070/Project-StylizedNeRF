import os
import cv2
import json
import torch
import pickle
import numpy as np
from PIL import Image
# mpl.use('Agg')
import matplotlib.pyplot as plt



def json_read_rgbd(DepthImg_path, RgbImg_path, factor=1.):
    with open(DepthImg_path, 'r') as file:
        depth = np.array(json.load(file))
    rgb = Image.open(RgbImg_path).convert('RGB')
    w, h = rgb.size
    rgb = rgb.resize((int(w / factor), int(h / factor)))
    depth = cv2.resize(depth, (rgb.size[0], rgb.size[1]))
    rgb, depth = np.array(rgb, np.float32), np.array(depth, np.float32)
    return depth, rgb


def read_rgbd(DepthImg_path, RgbImg_path):
    depth_img = np.array(Image.open(DepthImg_path), np.float32)
    rgb_image = Image.open(RgbImg_path).convert('RGB')
    rgb_image = rgb_image.resize((depth_img.shape[1], depth_img.shape[0]))
    rgb_image = np.array(rgb_image, np.float32)
    return depth_img, rgb_image


def json_save_depth(path, depth):
    h, w = depth.shape[0], depth.shape[1]
    depth_list = []
    for i in range(h):
        depth_list.append(depth[i].reshape([-1]).tolist())
    with open(path, 'w') as file:
        json.dump(depth_list, file)


def write_obj(v, path, f=None):
    v = np.array(v)
    if v.shape[-1] == 3:
        str_v = [f"v {vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    else:
        str_v = [f"v {vv[0]} {vv[1]} {vv[2]} {vv[3]} {vv[4]} {vv[5]}\n" for vv in v]
    if f is not None:
        str_f = [f"f {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    else:
        str_f = []

    with open(path, 'w') as meshfile:
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')




def read_frame_pose(path):
    """Read frame information from json file"""
    """
        Input: 
            path: json path of frame. i.e. 'frame_00000.json'
        Output:
            projectionMatrix: (4*4 ndarray) matrix of projection matrix for clipping
            intrinsic: (3*3 ndarray) intrinsic matrix of camera
            cameraPose: (4*4 ndarray) matrix of camera pose
            time: (float) time of frame
            index: (int) index of frame
    """
    with open(path, 'r') as file:
        data = json.load(file)
        projectionMatrix = np.reshape(data['projectionMatrix'], [4, 4])
        intrinsic = np.reshape(data['intrinsics'], [3, 3])
        cameraPose = np.reshape(data['cameraPoseARFrame'], [4, 4])
        time = float(data['time'])
        index = int(data['frame_index'])
    return projectionMatrix, intrinsic, cameraPose, time, index


def json_read_camera_parameters2(path, printout=False):
    with open(path, 'r') as file:
        data = json.load(file)
        timeStamp = data['timeStamp']
        cameraEulerAngle = data['cameraEulerAngle']
        imageResolution = data['imageResolution']
        cameraTransform = np.reshape(data['cameraTransform'], [4, 4])
        cameraPos = data['cameraPos']
        cameraIntrinsics = np.reshape(data['cameraIntrinsics'], [3, 3])
        cameraView = np.reshape(data['cameraView'], [4, 4])
        cameraProjection = np.reshape(data['cameraProjection'], [4, 4])

    if printout:
        parameters = [timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection]
        names = ['timeStamp', 'cameraEulerAngle', 'imageResolution', 'cameraTransform', 'cameraPos', 'cameraIntrinsics', 'cameraView', 'cameraProjection']
        for i in range(len(parameters)):
            print('******************************************************************************************')
            print(names[i])
            print(parameters[i])
            print('******************************************************************************************')

    return timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection


def json_read_camera_parameters(path, printout=False):
    with open(path, 'r') as file:
        data = json.load(file)
        timeStamp = []
        cameraEulerAngle = []
        imageResolution = []
        cameraTransform = np.reshape(data['cameraTransform'], [4, 4])
        cameraPos = []
        cameraIntrinsics = np.reshape(data['cameraIntrinsics'], [3, 3])
        cameraView = []
        cameraProjection = []

    if printout:
        parameters = [timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection]
        names = ['timeStamp', 'cameraEulerAngle', 'imageResolution', 'cameraTransform', 'cameraPos', 'cameraIntrinsics', 'cameraView', 'cameraProjection']
        for i in range(len(parameters)):
            print('******************************************************************************************')
            print(names[i])
            print(parameters[i])
            print('******************************************************************************************')

    return timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection


def json_save_camera_parameters(path, cp, intr):
    timeStamp = []
    cameraEulerAngle = []
    imageResolution = []
    cameraTransform = np.reshape(cp, [-1]).tolist()
    cameraPos = []
    cameraIntrinsics = np.reshape(intr, [-1]).tolist()
    cameraView = []
    cameraProjection = []

    parameters = [timeStamp, cameraEulerAngle, imageResolution, cameraTransform, cameraPos, cameraIntrinsics, cameraView, cameraProjection]
    names = ['timeStamp', 'cameraEulerAngle', 'imageResolution', 'cameraTransform', 'cameraPos', 'cameraIntrinsics', 'cameraView', 'cameraProjection']
    save_dict = {}
    for i in range(len(parameters)):
            save_dict[names[i]] = parameters[i]
    with open(path, 'w') as file:
        json.dump(save_dict, file)



def save_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def normalize_cps(cps):
    cps = np.array(cps, dtype=np.float32)
    avg_center = min_line_dist_center(cps[:, :3, 3], cps[:, :3, 2])
    cps[:, :3, 3] -= avg_center
    dists = np.linalg.norm(cps[:, :3, 3], axis=-1)
    radius = 1.1 * np.max(dists) + 1e-5
    # Corresponding parameters change
    cps[:, :3, 3] /= radius
    return cps, radius


def min_line_dist_center(rays_o, rays_d):
    if len(np.shape(rays_d)) == 2:
        rays_o = rays_o[..., np.newaxis]
        rays_d = rays_d[..., np.newaxis]
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((A_i @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist


def save_obj(path, obj):
    file = open(path, 'wb')
    obj_str = pickle.dumps(obj)
    file.write(obj_str)
    file.close()


def load_obj(path):
    file = open(path, 'rb')
    obj = pickle.loads(file.read())
    file.close()
    return obj


class plot_chart:
    def __init__(self, name='image', path='./plotting/', x_label='iter', y_label='Loss', max_len=100000):
        self.name = name
        self.path = path
        self.x_label = x_label
        self.y_label = y_label
        self.max_len = max_len
        self.ys, self.xs = None, None
        self.path = './chart'

    def draw(self, y, x):
        self.ys = np.array([y]) if self.ys is None else np.concatenate([self.ys, [y]])
        self.xs = np.array([x]) if self.xs is None else np.concatenate([self.xs, [x]])

        self.check_len()

        plt.close('all')
        plt.plot(self.xs, self.ys, "b.-")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        plt.savefig(self.path + "/" + self.name + ".png")

        self.save()

    def check_len(self):
        if self.ys.shape[0] > self.max_len:
            half_ids = np.arange(self.ys.shape[0]//2, self.ys.shape[0])
            self.ys = self.ys[half_ids]
            self.xs = self.xs[half_ids]

    def save(self):
        save_obj(self.path + '/chart_obj', self)




def dep2pcl(depth, intr, cp, pixel_alignment=True):
    intr = intr.copy()
    h, w = np.shape(depth)[:2]
    if pixel_alignment:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32) - 0.5, np.arange(h, dtype=np.float32) - 0.5, indexing='xy')
    else:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    z = - depth
    uvz = np.stack([u*z, v*z, z], axis=-1).reshape([-1, 3])
    # The z axis is toward the camera and y axis should be conversed
    intr[0, 0] = - intr[0, 0]
    intr_inverse = np.linalg.inv(intr)
    xyz_camera = np.einsum('bu,cu->bc', uvz, intr_inverse)
    xyz_camera = np.concatenate([xyz_camera, np.ones([xyz_camera.shape[0], 1])], axis=-1)
    xyz_world = np.einsum('bc,wc->bw', xyz_camera, cp)
    return xyz_world


def get_cos_map(h, w, cx, cy, f):
    i, j = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/f, -(j-cy)/f, -np.ones_like(i)], -1)
    cos = 1 / np.linalg.norm(dirs, axis=-1)
    cos = np.array(cos, dtype=np.float32)
    return cos


def pts2imgcoor(pts, intr):
    intr = intr.copy()
    intr[0, 0] *= -1
    imgcoor = np.einsum('bc,ic->bi', pts, intr)
    imgcoor /= imgcoor[..., -1][..., np.newaxis]
    return imgcoor


img2mse = lambda x, y: torch.mean((x - y) ** 2)
img2l1 = lambda x, y: (x - y).abs().mean()
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (256*np.clip(x,0,1)).astype(np.uint8)


def empty_loss(ts, sigma, t_gt):
    """Empty Loss"""
    """
    ts: [ray, N]
    sigma: [ray, N]
    t_gt: [ray]
    """
    delta_ts = ts[:, 1:] - ts[:, :-1]  # [ray, N-1]
    sigma = torch.relu(sigma[:, :-1])  # [ray, N-1]
    boarder_rate = 0.9
    sigma_sum = torch.sum(sigma * delta_ts * (ts[:, :-1] < (t_gt.unsqueeze(-1) * boarder_rate)).float(), dim=-1)
    loss_empty = torch.mean(sigma_sum)
    return loss_empty
