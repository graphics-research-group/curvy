# Curvy

In this work, we present a novel approach for reconstructing shape point clouds using planar sparse cross-sections with the help of generative modeling. We present unique challenges pertaining to the representation and reconstruction in this problem setting. Most methods in the classical literature lack the ability to generalize based on object class and employ complex mathematical machinery to reconstruct reliable surfaces. We present a simple learnable approach to generate a large number of points from a small number of input cross-sections over a large dataset. We use a compact parametric polyline representation using adaptive splitting to represent the cross-sections and perform learning using a Graph Neural Network to reconstruct the underlying shape in an adaptive manner reducing the dependence on the number of cross-sections provided.

This repository contains scripts for training the Curvy network.

## Contents

- `data_preprocessing.py`: Script for preprocessing the dataset.
- `vae_model.py`: Script defining the VAE model architecture.
- `train_vae.py`: Script for training the VAE model.
- `reconstruct.py`: Script for reconstructing images using the trained VAE model.
- `utils.py`: Utility functions used across various scripts.

## Requirements

- Python 3.9>=
- PyTorch
- PyTorchGeometric
- NumPy
- Matplotlib
- trimesh
- pandas

## Usage

1. **Data Preprocessing**:
    To generate cross-section data download ShapeNetCore.v2 dataset provided [https://shapenet.org/], set the paths in lines 574 where `root` refers to the path where ShapeNet dataset is downloaded and `target` is the path where the generated data will be stored in `Data_Prep.py` and run
    
    ```
    python Data_Prep.py
    ```

2. **Train AE**:
    Download data from the original PointNet repository [https://github.com/charlesq34/pointnet]
    ```
    python ae_paper.py --data_dir /path/to/preprocessed_data 
    ```

3. **Train GCN**:
    To train the GCN set the following paths - 
    - line 962: path to shapenet dataset
    - line 963: path to the generate cross-section data
    - lines 1011-1012 : the path for the autoencoder checkpoint (note: the same path would be used for loading both the encoder and decoder models)
    
    then run -
    ```
    python gcn.py

    ```
## Checkpoints
The trained checkpoints can be downloaded from 
[Google Drive](https://drive.google.com/drive/folders/1dCDAOEpV8SiGM5GKgN_ANTOMuvd4DjvO?usp=sharing)
The weights consist of autoencoder weights (`new_test_dict_ae_last.pth`) and gcn weights (`new_test_dict_gen_best_128.pth`)

## Note
To load the existing checkpoints on newer pytorch version the following snippet would be needed for version compatibility for newer pytorch geoemetric version for the GCN checkpoint
```
## for newer version of pytorch geometric
import torch

# Load the state dictionary
state_dict = torch.load(/path/to/checkpoint)

# Rename the keys for gat_conv1 and gat_conv2 layers
new_state_dict = {}
for key in state_dict.keys():
    if 'gat_conv1' in key or 'gat_conv2' in key:
        new_key = key.replace('att_l', 'att_src').replace('att_r', 'att_dst').replace('lin_l', 'lin').replace('lin_r', 'lin')
        new_state_dict[new_key] = state_dict[key]
    else:
        new_state_dict[key] = state_dict[key]

# Load the modified state dictionary into the model
gcn.load_state_dict(new_state_dict)
gcn.eval()
```
## License

This project is licensed under the MIT License.

