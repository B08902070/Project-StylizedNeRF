import numpy as np
import torch
import os
import platform


class Camera:
    def __init__(self, projectionMatrix=None, cameraPose=None, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.tensor_list = ['projectionMatrix', 'cameraPose', 'w2c_matrix']
        for attr in self.tensor_list:
            setattr(self, attr, None)
        self.set(projectionMatrix=projectionMatrix, cameraPose=cameraPose)

    def set(self, **kwargs):
        keys = kwargs.keys()
        func_map = {'projectionMatrix': self.set_project, 'cameraPose': self.set_pose}
        for name in keys:
            try:
                if name in func_map.keys():
                    func_map[name](kwargs[name])
                else:
                    raise ValueError(name + f'is not in{keys}')
            except ValueError as e:
                print(repr(e))

    def set_pose(self, cameraPose):
        if cameraPose is None:
            self.cameraPose = self.w2c_matrix = None
            return
        elif type(cameraPose) is np.ndarray:
            cameraPose = torch.from_numpy(cameraPose)
        self.cameraPose = cameraPose.float()
        self.w2c_matrix = torch.inverse(self.cameraPose).float()
        self.to(self.device)

    def set_project(self, projectionMatrix):
        if projectionMatrix is None:
            self.projectionMatrix = None
            return
        elif type(projectionMatrix) is np.ndarray:
            projectionMatrix = torch.from_numpy(projectionMatrix)
        self.projectionMatrix = projectionMatrix.float()
        self.to(self.device)

    def to(self, device):
        if type(device) is str:
            device = torch.device(device)
        self.device = device
        for tensor in self.tensor_list:
            if getattr(self, tensor) is not None:
                setattr(self, tensor, getattr(self, tensor).to(self.device))
        return self

    def WorldtoCamera(self, coor_world):
        coor_world = coor_world.clone()
        if len(coor_world.shape) == 2:
            coor_world = torch.cat([coor_world, torch.ones([coor_world.shape[0], 1]).to(self.device)], -1)
            coor_camera = torch.einsum('bcw,nw->bnc', self.w2c_matrix, coor_world)
        else:
            coor_world = self.homogeneous(coor_world)
            coor_camera = torch.einsum('bcw,bnw->bnc', self.w2c_matrix, coor_world)
        return coor_camera

    def CameratoWorld(self, coor_camera):
        coor_camera = coor_camera.clone()
        coor_camera = self.homogeneous(coor_camera)
        coor_world = torch.einsum('bwc,bnc->bnw', self.cameraPose, coor_camera)[:, :, :3]
        return coor_world

    def WorldtoCVV(self, coor_world):
        coor_camera = self.WorldtoCamera(coor_world)
        coor_cvv = torch.einsum('vc,bnc->bnv', self.projectionMatrix, coor_camera)
        coor_cvv = coor_cvv[..., :-1] / coor_cvv[..., -1:]
        return coor_cvv

    def homogeneous(self, coor3d, force=False):
        if coor3d.shape[-1] == 3 or force:
            coor3d = torch.cat([coor3d, torch.ones_like(coor3d[..., :1]).to(self.device)], -1)
        return coor3d

    def rasterize(self, coor_world, rgb, h=192, w=256, k=1.5, z=1):
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer import compositing
        from pytorch3d.renderer.points import rasterize_points

        def PixeltoCvv(h, w, hid=0, wid=0):
            cvv = torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]]).float()
            pts = Pointclouds(points=cvv, features=cvv)
            idx, _, dist2 = rasterize_points(pts, [h, w], 1e10, 3)
            a2, b2, c2 = (dist2.cpu().numpy())[0, hid, wid]
            x2 = (a2 + b2) / 2 - 1
            cosa = (x2 + 1 - a2) / (2 * x2**0.5)
            sina_abs = (1 - cosa**2)**0.5
            u = (x2 ** 0.5) * cosa
            v = (x2 ** 0.5) * sina_abs
            if np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5) > 1e-5:
                v = - (x2 ** 0.5) * sina_abs
                if(np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5) > 1e-5):
                    print(np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5), ' is too large...')
                    print(f"Found pixel {[hid, wid]} has uv: {(u, v)} But something wrong !!!")
                    print(f"a: {a2**0.5}, b: {b2**0.5}, c: {c2**0.5}, idx: {idx[0, 0, 0]}, dist2: {dist2[0, 0, 0]}")
                    os.exit(-1)
            return u, v

        batch_size = self.cameraPose.shape[0]
        point_num = rgb.shape[-2]
        coor_cvv = self.WorldtoCVV(coor_world).reshape([batch_size, point_num, 3])  # (batch_size, point, 3)
        umax, vmax = PixeltoCvv(h=h, w=w, hid=0, wid=0)
        umin, vmin = PixeltoCvv(h=h, w=w, hid=h-1, wid=w-1)
        coor_cvv[..., 0] = (coor_cvv[..., 0] + 1) / 2 * (umax - umin) + umin
        coor_cvv[..., 1] = (coor_cvv[..., 1] + 1) / 2 * (vmax - vmin) + vmin

        rgb = rgb.reshape([1, point_num, rgb.shape[-1]])  # (1, point, 3)
        rgb_coor = torch.cat([rgb, coor_world.unsqueeze(0)], dim=-1).expand([batch_size, point_num, 6])  # (1, point, 6)

        if platform.system() == 'Windows':
            # Bug of pytorch3D on windows
            hw = np.array([h, w])
            mindim, maxdim = np.argmin(hw), np.argmax(hw)
            aspect_ration = hw[maxdim] / hw[mindim]
            coor_cvv[:, :, mindim] *= aspect_ration

        pts3D = Pointclouds(points=coor_cvv, features=rgb_coor)
        radius = float(2. / max(w, h) * k)
        idx, _, _ = rasterize_points(pts3D, [h, w], radius, z)
        alphas = torch.ones_like(idx.float())
        img = compositing.alpha_composite(
            idx.permute(0, 3, 1, 2).long(),
            alphas.permute(0, 3, 1, 2),
            pts3D.features_packed().permute(1, 0),
        )
        img = img.permute([0, 2, 3, 1]).contiguous()  # (batch, h, w, 6)
        rgb_map, coor_map = img[..., :3], img[..., 3:]  # (batch, h, w, 3)
        msk = (idx[:, :, :, :1] != -1).float()  # (batch, h, w, 1)

        return rgb_map, coor_map, msk

    def rasterize_pyramid(self, coor_world, rgb, density=None, h=192, w=256, k=np.array([0.7, 1.2, 1.7, 2.2])):
        if density is None:
            density = torch.ones(coor_world.shape[0], 1).to(coor_world.device)
        mask = None
        image = None
        for ksize in k:
            img, _, msk = self.rasterize(coor_world, rgb, h, w, ksize, 10)
            mask = msk if mask is None else mask * msk
            image = img if image is None else image + img * mask.unsqueeze(-1).expand(img.shape)
        return image, mask
