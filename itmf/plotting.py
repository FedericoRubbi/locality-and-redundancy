import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import os

def plot_layerwise_test_train_metrics(history, epochs, depth, dataset_name, max_lines=32, save_dir='.'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Layer-wise Train/Test Metrics Evolution - {dataset_name}", fontsize=16)
    
    metrics = [
        ('layer_train_acc', 'Train Accuracy (%)', 'Blues', axes[0, 0]),
        ('layer_test_acc', 'Test Accuracy (%)', 'Greens', axes[0, 1]),
        ('layer_train_loss', 'Train Loss', 'Oranges', axes[1, 0]),
        ('layer_test_loss', 'Test Loss', 'Purples', axes[1, 1])
    ]
    
    indices = range(depth) if depth <= max_lines else np.linspace(0, depth - 1, max_lines, dtype=int).tolist()
    norm = mcolors.Normalize(vmin=0, vmax=depth-1)
    
    for key, title, cmap_name, ax in metrics:
        data = np.array(history[key])
        cmap = plt.get_cmap(cmap_name)
        
        for layer_idx in indices:
            color = cmap(0.4 + 0.6 * norm(layer_idx))
            ax.plot(range(1, epochs + 1), data[:, layer_idx], color=color, alpha=0.8)
            
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epochs")
        ax.grid(True, alpha=0.3)
        if 'loss' in key:
            ax.set_yscale('log')
        
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Layer Depth (Shallow -> Deep)", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"layerwise_train_test_metrics_{dataset_name.lower()}.png"))
    plt.close()

def plot_global_metrics_evolution(history, epochs, dataset_name, save_dir='.'):
    epochs_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Global Metrics Evolution - {dataset_name}", fontsize=16)
    
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='blue', lw=2)
    axes[0].plot(epochs_range, history['test_loss'], label='Test Loss', color='red', lw=2, linestyle='--')
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(epochs_range, history['train_acc'], label='Train Acc', color='blue', lw=2)
    axes[1].plot(epochs_range, history['test_acc'], label='Test Acc', color='red', lw=2, linestyle='--')
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"global_metrics_evolution_{dataset_name.lower()}.png"))
    plt.close()

def plot_standard_metrics(history, depth, dataset_name, epochs, save_dir='.'):
    layers_range = range(1, depth + 1)
    def get_final(hist, key): return hist[key][-1]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12)) 
    fig.suptitle(f"Local Bottleneck Final Metrics - {epochs} Epochs - {dataset_name}", fontsize=16)
    
    ax1 = axes[0,0]
    ax1.set_title("Classification: Test Accuracy vs CE")
    ax1.plot(layers_range, get_final(history, 'layer_test_acc'), 'o-', c='blue', label='Test Acc')
    ax1.set_ylabel("Accuracy (%)", color='blue')
    ax1b = ax1.twinx()
    ax1b.plot(layers_range, get_final(history, 'layer_test_loss'), 'o-', c='red', label='Test CE')
    ax1b.set_ylabel("Test CE Loss", color='red')
    
    ax2 = axes[0,1]
    ax2.set_title("Weight Orthogonality Penalty")
    ax2.plot(layers_range, get_final(history, 'ortho'), 'o-', color='brown')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0,2]
    ax3.set_title("Embedding Magnitude (L2 Norm)")
    ax3.plot(layers_range, get_final(history, 'mag_mean'), 'o-', c='darkorange')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1,0]
    ax4.set_title("Renyi Entropy (H2)")
    ax4.plot(layers_range, get_final(history, 'ent_mean'), 'o-', c='green')
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1,1]
    ax5.set_title("Local Reconstruction Loss (MSE)")
    ax5.plot(layers_range, get_final(history, 'layer_rec_loss'), 'o-', color='teal')
    ax5.set_ylabel("MSE Loss")
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1,2]
    ax6.set_title("Scheduled Dirichlet Concentration ($\\alpha_i$)")
    ax6.plot(layers_range, get_final(history, 'alphas'), 'o-', color='purple')
    ax6.set_ylabel("Concentration Parameter")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'standard_metrics_{dataset_name.lower()}.png'))
    plt.close()

def plot_alpha_progression(history, depth, dataset_name, save_dir='.'):
    alphas = history['alphas'][0]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, depth + 1), alphas, 'o-', color='purple', lw=2)
    plt.title(f"Dirichlet Target Schedule (Cosine) - {dataset_name}")
    plt.xlabel("Block Depth")
    plt.ylabel("Concentration $\\alpha_i$ (Higher = Sharper Target)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"alpha_schedule_{dataset_name.lower()}.png"))
    plt.close()

def plot_all_metrics_evolution(history, epochs, depth, dataset_name, max_lines=10, save_dir='.'):
    metrics_to_plot = [
        ('layer_train_acc', 'Layer Train Accuracy (%)'),
        ('layer_train_loss', 'Layer Train Loss (Soft CE)'),
        ('layer_rec_loss', 'Layer Reconstruction Loss (MSE)'),
        ('ortho', 'Orthogonality Penalty'), 
        ('ent_mean', 'Renyi Entropy (H2)'), 
        ('mag_mean', 'Embedding Magnitude')
    ]
    
    indices = list(range(depth)) if depth <= max_lines else np.linspace(0, depth - 1, max_lines, dtype=int).tolist()
    rows = (len(metrics_to_plot) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    fig.suptitle(f"Evolution of Layer-wise Metrics - {dataset_name}", fontsize=18, y=1.02)
    axes = axes.flatten()

    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=depth-1)

    for idx, (key, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        data = np.array(history[key]) 
        
        for layer_idx in indices:
            color = cmap(norm(layer_idx))
            ax.plot(range(1, epochs + 1), data[:, layer_idx], color=color, alpha=0.7)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epochs")
        ax.grid(True, alpha=0.2)
        if key in ['ortho', 'layer_train_loss', 'layer_rec_loss']: ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label("Layer Depth (Shallow -> Deep)", rotation=270, labelpad=20, fontsize=14)
    
    plt.savefig(os.path.join(save_dir, f"metrics_evolution_{dataset_name.lower()}.png"))
    plt.close()