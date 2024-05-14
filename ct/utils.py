import os
from pymatgen.core.structure import Structure
from tqdm import tqdm
import numpy as np
import json
import torch
def list_space_groups(directory, save_file="space_groups.json"):
    # Check if the space groups file already exists
    if os.path.exists(save_file):
        with open(save_file, 'r') as file:
            space_groups = json.load(file)
        return space_groups

    space_groups = set()

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".cif"):
            file_path = os.path.join(directory, filename)

            try:
                # 读取 CIF 文件
                structure = Structure.from_file(file_path)
                # 获取空间群
                space_group = structure.get_space_group_info()[0]
                # 添加到集合中
                space_groups.add(space_group)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save the space groups list to a file
    with open(save_file, 'w') as file:
        json.dump(list(space_groups), file)

    return list(space_groups)

def get_cif_files(directory):
    cif_files = []

    # 确保目录存在
    assert os.path.exists(directory), f"{directory} does not exist!"

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".cif"):
            file_path = os.path.join(directory, filename)
            cif_files.append(file_path)

    return cif_files

def extract_cif_id(cif_string):
    # Split the string on backslash and take the last part
    parts = cif_string.split('\\')
    if parts:
        # Take the last part and split on dot, then take the first part which should be the ID
        id_part = parts[-1].split('.')[0]
        return id_part
    else:
        return "No ID found"
    
def space_group_to_onehot(space_group, space_group_to_index):
    # 创建一个零向量，长度与空间群列表一致
    onehot = np.zeros(len(space_group_to_index))
    # 设置对应空间群索引的位置为 1
    onehot[space_group_to_index[space_group]] = 1
    return onehot

def get_lattice_parameters(structure):
    lattice = structure.lattice
    # 获取三个边长
    a, b, c = lattice.a, lattice.b, lattice.c
    # 获取三个角度
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    
    return a, b, c, alpha, beta, gamma
from sklearn.metrics import r2_score
import numpy as np
def calculate_r2_per_feature(target, prediction):
    """
    Calculate the R2 score for each feature in the batch.

    :param target: Target tensor of shape [batch, c].
    :param prediction: Prediction tensor of shape [batch, c].
    :return: List of R2 scores, one for each feature.
    """
    r2_scores = []
    target_np = target
    prediction_np = prediction

    # Calculate R2 score for each feature
    for i in range(target_np.shape[1]):  # Iterate over features
        score = r2_score(target_np[:, i], prediction_np[:, i])
        r2_scores.append(score)

    return r2_scores

def format_list_to_string(lst):
    return '[' + ', '.join(f'{x:.3f}' for x in lst) + ']'



def calculate_accuracy_within_margin_percent(target, output, margin_percent):
    """
    Calculate the percentage of predictions that are within a specified margin percent of the target values.
    Both target and output are [batch, c] where c is not 1.

    :param target: Ground truth values, a 2D numpy array or a 2D PyTorch tensor.
    :param output: Predicted values, a 2D numpy array or a 2D PyTorch tensor.
    :param margin_percent: The margin percent within which the predictions are considered accurate.
    :return: A list containing the accuracy for each feature within the specified margin.
    """
    if not isinstance(target, np.ndarray):
        target = target.cpu().numpy()
    if not isinstance(output, np.ndarray):
        output = output.cpu().numpy()

    # Calculate the margin
    margin = np.abs(margin_percent / 100.0 * target)
    # Calculate whether each prediction is within the margin
    within_margin = np.abs(output - target) <= margin
    # Calculate the accuracy for each feature
    accuracy_per_feature = np.mean(within_margin, axis=0)

    return accuracy_per_feature

def weighted_mse_loss(output, target, weights=[1,1,1,1,1,1]):
    # 计算每个特征的MSE
    weights=weights.to(output.device)
    se = (output - target) ** 2
    # 应用权重
    weighted_se = se * weights
    # 计算加权平均
    weighted_mse = weighted_se.mean()
    return weighted_mse

def shuffle_tensor_along_dim(tensor, dim):
    """
    Shuffle a tensor along a specified dimension.

    :param tensor: Input tensor.
    :param dim: Dimension along which to shuffle.
    :return: Shuffled tensor.
    """
    # Get the size of the specified dimension
    dim_size = tensor.size(dim)
    # Generate a random permutation of indices along the specified dimension
    idx = torch.randperm(dim_size).to(tensor.device)
    # Shuffle the tensor along the specified dimension using the generated indices
    shuffled_tensor = tensor.index_select(dim, idx)
    return shuffled_tensor