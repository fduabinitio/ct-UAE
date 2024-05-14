from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from scipy.spatial.distance import pdist, squareform



def get_train_val_test_loader(dataset, collate_fn=default_collate,collate_fn_train=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn_train, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):

    N_max = max(atom_fea.shape[0] for (atom_fea, _), _,_,_,_, _ in dataset_list)

    # 初始化批处理列表和掩码列表
    batch_atom_fea, batch_coords, batch_mask = [], [], []
    batch_target1 = []
    batch_target2 = []
    batch_target3 = []
    batch_target4 = []
    batch_cif_ids = []

    for (atom_fea, atom_coords), target1,target2, target3,target4,cif_id in dataset_list:
        n_i = atom_fea.shape[0]  # 当前晶体的原子数
        # 创建填充（pad）后的特征和坐标
        pad_atom_fea = np.pad(atom_fea, ((0, N_max - n_i), (0, 0)), mode='constant', constant_values=0)
        pad_atom_coords = np.pad(atom_coords, ((0, N_max - n_i), (0, 0)), mode='constant', constant_values=0)
        # 创建掩码
        mask = np.zeros((N_max, 1), dtype=int)
        mask[:n_i] = 1  # 真实原子位置设置为1

        # 添加到批处理列表
        batch_atom_fea.append(pad_atom_fea)
        batch_coords.append(pad_atom_coords)
        batch_mask.append(mask)
        batch_target1.append(target1)
        batch_target2.append(target2)
        batch_target3.append(target3)
        batch_target4.append(target4)
        batch_cif_ids.append(cif_id)

    # 转换为张量
    batch_atom_fea = torch.FloatTensor(batch_atom_fea)
    batch_coords = torch.FloatTensor(batch_coords)
    batch_mask = torch.ByteTensor(batch_mask)  # 使用ByteTensor来表示掩码
    batch_target1 = torch.FloatTensor(batch_target1)
    batch_target2 = torch.FloatTensor(batch_target2)
    batch_target3 = torch.FloatTensor(batch_target3)
    batch_target4 = torch.FloatTensor(batch_target4)
    batch_mask = batch_mask.bool()
    batch_mask = ~batch_mask.squeeze(-1)
    return (batch_atom_fea, batch_coords, batch_mask), batch_target1,batch_target2,batch_target3,batch_target4, batch_cif_ids



def collate_pool_train(dataset_list):
    N_max = max(atom_fea.shape[0] for (atom_fea, _), _,_,_,_, _ in dataset_list)

    # 初始化批处理列表和掩码列表
    batch_atom_fea, batch_coords, batch_mask = [], [], []
    batch_target1 = []
    batch_target2 = []
    batch_target3 = []
    batch_target4 = []
    batch_cif_ids = []

    for (atom_fea, atom_coords), target1,target2, target3,target4,cif_id in dataset_list:
        n_i = atom_fea.shape[0]  # 当前晶体的原子数
        if random.random() < 0.5:
            atom_fea += torch.randn_like(atom_fea) * 0.001
            atom_coords += torch.randn_like(atom_coords) * 0.001
        # 创建填充（pad）后的特征和坐标
        pad_atom_fea = np.pad(atom_fea, ((0, N_max - n_i), (0, 0)), mode='constant', constant_values=0)
        pad_atom_coords = np.pad(atom_coords, ((0, N_max - n_i), (0, 0)), mode='constant', constant_values=0)
        # 创建掩码
        mask = np.zeros((N_max, 1), dtype=int)
        mask[:n_i] = 1  # 真实原子位置设置为1

        # 添加到批处理列表
        batch_atom_fea.append(pad_atom_fea)
        batch_coords.append(pad_atom_coords)
        batch_mask.append(mask)
        batch_target1.append(target1)
        batch_target2.append(target2)
        batch_target3.append(target3)
        batch_target4.append(target4)
        batch_cif_ids.append(cif_id)

    # 转换为张量
    batch_atom_fea = torch.FloatTensor(batch_atom_fea)
    batch_coords = torch.FloatTensor(batch_coords)
    batch_mask = torch.ByteTensor(batch_mask)  # 使用ByteTensor来表示掩码
    batch_target1 = torch.FloatTensor(batch_target1)
    batch_target2 = torch.FloatTensor(batch_target2)
    batch_target3 = torch.FloatTensor(batch_target3)
    batch_target4 = torch.FloatTensor(batch_target4)
    batch_mask = batch_mask.bool()
    batch_mask = ~batch_mask.squeeze(-1)
    return (batch_atom_fea, batch_coords, batch_mask), batch_target1,batch_target2, batch_target3,batch_target4,batch_cif_ids

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):

    def __init__(self, root_dir,file='id_prop.csv', max_num_nbr=12, radius=16, dmin=0, step=0.4 ,bins=64,
                 random_seed=42):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, file)
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
    def __len__(self):
        return len(self.id_prop_data)

    def _augment_coordinates(self, coords, max_translation=5, max_rotation=np.pi):
        """
        Apply random rotation, translation, and reflection to the coordinates.

        :param coords: numpy array of shape (N, 3) representing atomic coordinates.
        :param max_translation: float representing maximum translation.
        :param max_rotation: float representing maximum rotation angle in radians.
        :return: numpy array of augmented coordinates.
        """
        # Translation


        translation_vector = np.random.uniform(-max_translation, max_translation, coords.shape[1])
        trans_coords = coords + translation_vector

        # Rotation

        rotation_angle_x = np.random.uniform(-max_rotation, max_rotation)
        rotation_angle_y = np.random.uniform(-max_rotation, max_rotation)
        rotation_angle_z = np.random.uniform(-max_rotation, max_rotation)

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rotation_angle_x), -np.sin(rotation_angle_x)],
                       [0, np.sin(rotation_angle_x), np.cos(rotation_angle_x)]])

        Ry = np.array([[np.cos(rotation_angle_y), 0, np.sin(rotation_angle_y)],
                       [0, 1, 0],
                       [-np.sin(rotation_angle_y), 0, np.cos(rotation_angle_y)]])

        Rz = np.array([[np.cos(rotation_angle_z), -np.sin(rotation_angle_z), 0],
                       [np.sin(rotation_angle_z), np.cos(rotation_angle_z), 0],
                       [0, 0, 1]])

        rotation_matrix = Rz @ Ry @ Rx
        rotate_coords = np.dot(trans_coords, rotation_matrix)

        # Reflection

        axis = random.choice(['x', 'y', 'z'])
        if axis == 'x':
            reflection_matrix = np.array([[-1, 0, 0],
                                          [0, 1, 1],
                                          [0, 0, 1]])
        elif axis == 'y':
            reflection_matrix = np.array([[1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, 1]])
        else:  # Reflection over z axis
            reflection_matrix = np.array([[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, -1]])
        ref_coords = np.dot(rotate_coords, reflection_matrix)

        return ref_coords

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target1,target2,target3,target4 = self.id_prop_data[idx]
        try:
            crystal = Structure.from_file(os.path.join(self.root_dir,
                                                        cif_id + '.cif'))
        except:
            print(cif_id)
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        #all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        #all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        atom_fea = torch.Tensor(atom_fea)
        atom_coords = np.array([site.coords for site in crystal.sites])
        if random.random()<0.5:
            atom_coords = self._augment_coordinates(atom_coords)
        atom_coords = torch.Tensor(atom_coords)
        target1 = torch.tensor([float(target1)], dtype=torch.float)
        target2 = torch.tensor([float(target2)], dtype=torch.float)
        target3 = torch.tensor([float(target3)], dtype=torch.float)
        target4 = torch.tensor([float(target4)], dtype=torch.float)

        return (atom_fea, atom_coords), target1,target2,target3,target4, cif_id

