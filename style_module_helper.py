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


def default_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform=None):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        transform = default_transform() if transform is None else transform
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


class CoorImageDataset(data.Dataset):
    def __init__(self, root, transform=default_transform()):
        super(CoorImageDataset, self).__init__()
        self.root = root
        self.image_paths = sorted(list(Path(self.root).glob('rgb_*.png')))
        self.geo_paths = sorted(list(Path(self.root).glob('geometry_*.npz')))
        data = np.load(str(self.geo_paths[0]))
        self.hwf = data['hwf']
        # self.near, self.far = data['near'], data['far']
        self.near, self.far = 0., 1.
        self.transform = transform

    def __getitem__(self, index):
        image_path, geo_path = self.image_paths[index], self.geo_paths[index]
        img = Image.open(str(image_path)).convert('RGB')
        img = self.transform(img)
        geo = np.load(str(geo_path))
        coor_map, cps = geo['coor_map'], geo['cps']
        return img, coor_map, cps

    def __len__(self):
        return len(self.image_paths)

    def name(self):
        return 'FlatFolderDataset'


class CoorImageDataset_pl(data.Dataset):
    def __init__(self, root, factor=0.01):
        super(CoorImageDataset_pl, self).__init__()
        self.root = root
        self.image_paths = sorted(list(Path(self.root).glob('rgb_*.png')))
        self.geo_paths = sorted(list(Path(self.root).glob('geometry_*.npz')))
        data = np.load(str(self.geo_paths[0]))
        self.hwf = data['hwf']
        # self.near, self.far = data['near'], data['far']
        self.near, self.far = 0., 1.
        self.factor = factor
        self.transform = default_transform()

        ts = np.zeros([len(self.geo_paths), 3], dtype=np.float32)
        for i in range(len(self.geo_paths)):
            ts[i] = np.load(str(self.geo_paths[i]))['cps'][:3, 3]

        dist = ts[np.newaxis] - ts[:, np.newaxis]
        dist = dist ** 2
        dist = dist.sum(-1) ** 0.5
        self.dist = dist

    def get_batch(self, batch_size, index=None):
        if index is None:
            index = np.random.randint(0, len(self.image_paths))
        dists = self.dist[index]
        inds = np.argsort(dists)
        prange = max(int(self.factor*len(self.image_paths)), batch_size)
        inds = inds[:prange]
        inds = np.random.choice(inds, [batch_size], replace=(prange <= batch_size))
        imgs, coor_maps, cps = [], [], []
        for i in range(batch_size):
            img, coor_map, cp = self.__getitem__(inds[i])
            imgs.append(img)
            coor_maps.append(coor_map)
            cps.append(cp)
        imgs = torch.stack(imgs).float()
        coor_maps = torch.from_numpy(np.stack(coor_maps)).float()
        cps = torch.from_numpy(np.stack(cps)).float()
        return imgs, coor_maps, cps

    def __getitem__(self, index):
        image_path, geo_path = self.image_paths[index], self.geo_paths[index]
        img = Image.open(str(image_path)).convert('RGB')
        img = self.transform(img)
        geo = np.load(str(geo_path))
        coor_map, cps = geo['coor_map'], geo['cps']
        return img, coor_map, cps

    def __len__(self):
        return len(self.image_paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(lr, lr_decay, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

