import torch
import numpy as np
import os
import multiprocessing as mp

def run_pca_umap_cuml(z_data, labels, epoch, save_dir):
    try:
        import cuml
        pca = cuml.PCA(n_components=50)
        z_pca = pca.fit_transform(z_data)
        
        umap = cuml.UMAP(n_components=2)
        z_umap = umap.fit_transform(z_pca)
        
        output_path = os.path.join(save_dir, f'umap_epoch_{epoch}.npy')
        np.save(output_path, {'umap': z_umap, 'labels': labels})
    except ImportError:
        print("cuml not found. Ensure RAPIDS environment is active for GPU accelerated UMAP.")

def async_visualize_callback(z_list, labels, epoch, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    z_cat = torch.stack(z_list).cpu().numpy()
    labels = labels.cpu().numpy()
    
    p = mp.Process(target=run_pca_umap_cuml, args=(z_cat.reshape(-1, z_cat.shape[-1]), labels, epoch, save_dir))
    p.start()
