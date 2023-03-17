"""utils.py: Utilites"""

import kornia 
import matplotlib.pyplot as plt
import torch

def plot_batch(batch:torch.Tensor, num_rows=1,imgsize=(2,2)):
    
    batch = kornia.utils.tensor_to_image(batch)

    # implicitly clip 
    batch = batch.clip(0,1) if batch.dtype == torch.float else batch.clip(0,255)

    n = batch.shape[0]
    num_cols = n // num_rows + bool(n % num_rows)
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*imgsize[0], num_rows*imgsize[1]))
    
    ax = ax.reshape(1,-1) if ax.ndim == 1 else ax # enforce 2dim

    for i in range(n):
        row = i // num_cols
        col = i % num_cols
        ax[row, col].imshow(batch[i])
        ax[row, col].axis("off")
    plt.tight_layout()
    plt.show()