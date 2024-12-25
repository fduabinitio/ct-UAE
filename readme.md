# ct-UAEs

A brief description of the ct-UAEs project, focusing on its objectives and the value it provides.

## Step 1: Download the Dataset

The datasets required for the ct-UAEs project are available via the following Google Drive links. Please download them as part of the initial setup:

- **mp_13 Dataset**: [Download here](https://drive.google.com/file/d/1u1n_CoPfVJVbtXr9Y8mONkv3hy6_fHTM/view?usp=sharing)
- **mp Dataset**: [Download here](https://drive.google.com/file/d/1RxDl48_MfMWpIMvgGcGX2pI9ZMusR6Dl/view?usp=drive_link)

Refer to the `download.txt` for additional guidance on downloading these datasets.

## Step 2: Extract and Organize Datasets

## Step 2: Extract and Organize Datasets

Once downloaded, extract each dataset to the project directory under their respective names (`mp` and `mp_13`). Ensure they are placed according to the following structure:
```csharp
ct-UAEs/
│
├── ...
├── embeddings/ # Corresponding ct-UAEs 
├── ct/ # Main source code directory
├── mp/ # Extracted mp dataset
├── mp_13/ # Extracted mp_13 dataset
├── examples/ # Example dataset folder
├── run/ # Folder for shell scripts
├── ...
└── download.txt
```
This organization is crucial for ensuring that the datasets are correctly recognized by the project scripts.

## Next Steps

- **Set up your development environment**: Ensure all necessary libraries and dependencies are installed. You can run 
```
conda create -n ct-UAEs python==3.10

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```
- **Begin model training**:execute the provided scripts to start training your models.

## How to Run

To train the models for single or multi-task learning, follow the steps below:

### Single-Task Training

For single-task training, execute the following command:

```bash
python train\\main.py --load mp_13 --file mp_13_1_f.csv
```
Replace mp_13 with mp and file with mp_formation.csv if you prefer to use the mp dataset. The --file flag specifies the CSV file containing the dataset. Here, f represents the formation energy (f) property.

### Multi-Task Training
For multi-task training with different combinations of properties, execute the corresponding script. For example, for formation energy (f) and bandgap (b), use:
```bash
python train\\main_mt_2.py --load mp_13 --file mp_13_2_fb.csv
```
For formation energy (f), bandgap (b), and total energy (e), use:
```bash
python train\\main_mt_3.py --load mp_13 --file mp_13_3_fbe.csv
```
For all four properties (formation energy (f), bandgap (b), total energy (e), and total magnetization (m)), use:
```bash 
python train\\main_mt_4.py --load mp_13 --file mp_13_4_fbem.csv
```
### Example Training
To train using the example dataset provided in the examples folder, use the following command:
```bash
python train/main.py --load examples --file id_prop.csv
```
## Running Training Scripts with Shell Scripts
All training commands can be executed using pre-defined shell scripts located in the run folder. For example:
### For single-task training on the mp dataset:
```bash
bash run/run_main_mp.sh
```
### For multi-task training on the mp dataset:
```bash
bash run/run_main_mt_2_mp.sh
bash run/run_main_mt_3_mp.sh
bash run/run_main_mt_4_mp.sh
```
### To run the example dataset:
```bash 
bash run/run_main.sh
```

## Universial Atomic Embeddings
The ct-UAEs project focuses on leveraging advanced machine learning techniques, particularly deep learning, to develop embeddings that are valuable for materials science research. The core objective of the project is to create accurate and useful representations (embeddings) of chemical compounds using a dataset of materials properties, which can then be used in various predictive models in materials informatics.
The embeddings themselves are stored in checkpoint files (.pth.tar), and extracting these embeddings involves loading specific layers from the trained models. Here's a guide on how to correctly load and utilize these model weights to obtain the embedding matrix:

### Updated Code to Load Embeddings from Checkpoints
The following markdown-formatted code snippet provides an improved and clear method to load the embedding weights from a given checkpoint (model_best_mt3_256.pth.tar). This ensures the extraction of a 100x128 tensor and 100 bias representing the embeddings:
```python
import torch
import os

def load_embeddings(checkpoint_path):
    """
    Loads the embedding weights and biases from a specified checkpoint file.

    Args:
    checkpoint_path (str): The path to the checkpoint file.

    Returns:
    dict: A dictionary containing the tensors for embedding weights and biases.
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Extract the embedding weights and biases
        embeddings = {name.replace('atom_embed.', ''): param
                      for name, param in checkpoint['state_dict'].items()
                      if 'atom_embed' in name}
        weights = embeddings.get('weight', None)
        biases = embeddings.get('bias', None)
        
        if weights is None or biases is None:
            raise ValueError("Weight or bias parameters not found in the checkpoint.")

        return {'weights': weights, 'biases': biases}

    else:
        raise FileNotFoundError("Checkpoint file not found.")

# Example Usage
checkpoint_path = 'embeddings\model_best_mt3_256.pth.tar'
embeddings = load_embeddings(checkpoint_path)
print("Loaded Weights Shape:", embeddings['weights'].shape)
print("Loaded Biases Shape:", embeddings['biases'].shape)

# load model ......
```

### Explanation
1. **Function Definition**: The function `load_embeddings` is designed to take a path to a checkpoint file.
2. **Checkpoint Loading**: It checks if the file exists and loads it. The `map_location` parameter is set to use CPU for compatibility.
3. **Embedding Extraction**: Extracts the specific layer weights that contain the embeddings (identified by the keyword 'atom_embed' in the state dictionary keys).
4. **Tensor Extraction**: The example usage at the bottom demonstrates how to call this function and print the shape of the loaded embeddings.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

<!-- ## License

This project is licensed under the MIT License. -->

## Acknowledgments

We acknowledge the contributors and datasets used in this project for their valuable contributions to scientific research.
