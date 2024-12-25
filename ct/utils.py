import os
from pymatgen.core.structure import Structure
from tqdm import tqdm
import numpy as np
import json
import torch
def list_space_groups(directory, save_file="space_groups.json"):
    if os.path.exists(save_file):
        with open(save_file, 'r') as file:
            space_groups = json.load(file)
        return space_groups

    space_groups = set()
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".cif"):
            file_path = os.path.join(directory, filename)

            try:
                
                structure = Structure.from_file(file_path)
                space_group = structure.get_space_group_info()[0]
                space_groups.add(space_group)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(save_file, 'w') as file:
        json.dump(list(space_groups), file)

    return list(space_groups)

def get_cif_files(directory):
    cif_files = []

    assert os.path.exists(directory), f"{directory} does not exist!"
    for filename in os.listdir(directory):
        if filename.endswith(".cif"):
            file_path = os.path.join(directory, filename)
            cif_files.append(file_path)

    return cif_files

def extract_cif_id(cif_string):
    parts = cif_string.split('\\')
    if parts:
        id_part = parts[-1].split('.')[0]
        return id_part
    else:
        return "No ID found"
    
def space_group_to_onehot(space_group, space_group_to_index):

    onehot = np.zeros(len(space_group_to_index))
    onehot[space_group_to_index[space_group]] = 1
    return onehot

def get_lattice_parameters(structure):
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
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

    for i in range(target_np.shape[1]):  
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

    
    margin = np.abs(margin_percent / 100.0 * target)
    within_margin = np.abs(output - target) <= margin
    accuracy_per_feature = np.mean(within_margin, axis=0)

    return accuracy_per_feature

def weighted_mse_loss(output, target, weights=[1,1,1,1,1,1]):
    weights=weights.to(output.device)
    se = (output - target) ** 2
    weighted_se = se * weights
    weighted_mse = weighted_se.mean()
    return weighted_mse

def shuffle_tensor_along_dim(tensor, dim):
    """
    Shuffle a tensor along a specified dimension.

    :param tensor: Input tensor.
    :param dim: Dimension along which to shuffle.
    :return: Shuffled tensor.
    """
    dim_size = tensor.size(dim)
    idx = torch.randperm(dim_size).to(tensor.device)
    shuffled_tensor = tensor.index_select(dim, idx)
    return shuffled_tensor