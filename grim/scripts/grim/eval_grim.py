import os
import sys
import torch
import torch.nn.functional as F
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

torch.set_float32_matmul_precision('high')

from data.data import get_ablation_dataloaders
from src.grim_models import Adapter, ConvAdapter, LightConvAdapter, ResNetAdapter, RecurrentEncoder, BidirectionalDecoder
from src.metrics import run_linear_probe


def compute_trajectories(adapter, encoder, loader, T, device, keep_depths, desc):
    keep = set(keep_depths)
    reprs = {t: [] for t in keep}
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc):
            v = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            v = v.to(device)

            h = adapter(v)
            if 0 in keep:
                reprs[0].append(h.cpu())
            all_labels.append(labels.cpu())

            for t in range(T):
                step_emb = torch.full((v.shape[0], 1), float(t), device=device)
                h = encoder(h, step_emb)
                if t + 1 in keep:
                    reprs[t + 1].append(h.cpu())

    reprs = {t: torch.cat(chunks, dim=0) for t, chunks in reprs.items()}
    return reprs, torch.cat(all_labels, dim=0)


def subsample_probe_set(X, y, max_samples):
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    idx = torch.randperm(X.shape[0])[:max_samples]
    return X[idx], y[idx]


def compute_reconstruction_metrics(adapter, encoder, decoder, test_loader, T, device, local_depths):
    local_depths = set(local_depths)
    mse_total = 0.0
    num_samples = 0
    trajectory_diffs = {t: 0.0 for t in local_depths}

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Test Backward Trajectories"):
            N = inputs.shape[0]
            v = inputs.to(device)
            h0 = adapter(v)

            fwd_history = [h0]
            curr_h = h0
            for t in range(T):
                step_emb = torch.full((N, 1), float(t), device=device)
                curr_h = encoder(curr_h, step_emb)
                fwd_history.append(curr_h)

            rev_h = fwd_history[-1]
            for t in reversed(range(T)):
                step_emb = torch.full((N, 1), float(t), device=device)
                rev_h = decoder(rev_h, step_emb)

                if t in local_depths:
                    trajectory_diffs[t] += F.mse_loss(rev_h, fwd_history[t]).item() * N

            mse_total += F.mse_loss(rev_h, h0, reduction='sum').item()
            num_samples += N

    local_mse = {t: trajectory_diffs[t] / num_samples for t in local_depths}
    return local_mse, mse_total / num_samples


def run_frontier_evaluation(adapter, encoder, decoder, test_loader, train_loader, T, device, linear_probe_config):
    adapter.eval()
    encoder.eval()
    max_samples = linear_probe_config.get('max_samples')

    print(f"\n--- Frontier Evaluation (depth {T}) ---")
    train_reprs, train_labels = compute_trajectories(
        adapter, encoder, train_loader, T, device, keep_depths=[T], desc="Train Frontier Trajectories")
    test_reprs, test_labels = compute_trajectories(
        adapter, encoder, test_loader, T, device, keep_depths=[T], desc="Test Frontier Trajectories")

    X_train, y_train = subsample_probe_set(train_reprs[T], train_labels, max_samples)
    acc = run_linear_probe(
        X_train, y_train, test_reprs[T], test_labels,
        epochs=linear_probe_config['epochs'],
        lr=linear_probe_config['lr'],
        batch_size=linear_probe_config['batch_size']
    )
    print(f"  Frontier probe accuracy at t={T}: {acc * 100:.2f}%")

    frontier_local_mse, final_mse = None, None
    if decoder is not None:
        decoder.eval()
        local_mse, final_mse = compute_reconstruction_metrics(
            adapter, encoder, decoder, test_loader, T, device, local_depths=[T - 1])
        frontier_local_mse = local_mse[T - 1]
        print(f"  Local MSE t={T} -> t={T-1}: {frontier_local_mse:.6f}")
        print(f"  Final Generation Reconstruction MSE: {final_mse:.6f}")

    return {
        "frontier_depth": T,
        "frontier_accuracy": acc,
        "frontier_local_mse": frontier_local_mse,
        "final_reconstruction_mse": final_mse,
    }


def run_evaluation(adapter, encoder, decoder, test_loader, train_loader, T, device, linear_probe_config):
    adapter.eval()
    encoder.eval()
    decoder.eval()
    max_samples = linear_probe_config.get('max_samples')

    print("\n--- Computing Forward Trajectories and Linear Probes ---")
    all_depths = list(range(T + 1))
    train_reprs, train_labels = compute_trajectories(
        adapter, encoder, train_loader, T, device, keep_depths=all_depths, desc="Train Forward Trajectories")
    test_reprs, test_labels = compute_trajectories(
        adapter, encoder, test_loader, T, device, keep_depths=all_depths, desc="Test Forward Trajectories")

    print("\nLinear Probe Accuracies across timesteps:")
    accuracy_vs_depth = []
    for t in range(T + 1):
        X_train, y_train = subsample_probe_set(train_reprs[t], train_labels, max_samples)
        acc = run_linear_probe(
            X_train, y_train, test_reprs[t], test_labels,
            epochs=linear_probe_config['epochs'],
            lr=linear_probe_config['lr'],
            batch_size=linear_probe_config['batch_size']
        )
        print(f"  t={t} | Accuracy: {acc * 100:.2f}%")
        accuracy_vs_depth.append(acc)

    print("\n--- Computing Backward Generation Trajectories ---")
    local_mse, final_mse = compute_reconstruction_metrics(
        adapter, encoder, decoder, test_loader, T, device, local_depths=range(T))

    print("\nDecoder Backward Trace Local MSE:")
    local_mse_vs_depth = [local_mse[t] for t in range(T)]
    for t in reversed(range(T)):
        print(f"  t={t+1} -> t={t} | Local MSE: {local_mse_vs_depth[t]:.6f}")

    print(f"\nFinal Generation Reconstruction MSE (Full trajectory): {final_mse:.6f}")

    print("\n--- Computing Elastic Probes ---")
    X_train, y_train = subsample_probe_set(train_reprs[T], train_labels, max_samples)
    max_depth_probe = run_linear_probe(
        X_train, y_train, test_reprs[T], test_labels,
        epochs=linear_probe_config['epochs'],
        lr=linear_probe_config['lr'],
        batch_size=linear_probe_config['batch_size'],
        return_model=True
    )

    elastic_accuracy_vs_depth = []
    for t in range(T + 1):
        elastic_acc = run_linear_probe(
            X_train, y_train, test_reprs[t], test_labels,
            model=max_depth_probe,
            epochs=linear_probe_config['epochs'],
            lr=linear_probe_config['lr'],
            batch_size=linear_probe_config['batch_size']
        )
        print(f"  Elastic Probe Accuracy at t={t} (trained on t={T}): {elastic_acc * 100:.2f}%")
        elastic_accuracy_vs_depth.append(elastic_acc)

    return {
        "depths": list(range(T + 1)),
        "accuracy_vs_depth": accuracy_vs_depth,
        "mse_depths": list(range(T)),
        "local_mse_vs_depth": local_mse_vs_depth,
        "elastic_accuracy_vs_depth": elastic_accuracy_vs_depth,
        "final_reconstruction_mse": final_mse,
    }


def eval_grim(config_path, eval_config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(eval_config_path, 'r') as f:
        eval_config = yaml.safe_load(f)

    run_name = config.get('run_name', 'mlp_baseline_0_drift')
    dataset = config['training'].get('dataset', 'mnist')
    num_workers = config['training'].get('num_workers', 4)
    batch_size = config['training']['batch_size']
    T = config['training']['T']

    model_cfg = config['model']
    delta = model_cfg['delta']
    embed_dim = model_cfg['embed_dim']
    adapter_type = model_cfg.get('adapter_type', 'mlp')
    adapter_hidden_dim = model_cfg.get('adapter_hidden_dim', 2048)
    patch_size = model_cfg.get('patch_size', 8)
    resnet_arch = model_cfg.get('resnet_arch', 'resnet50')
    resnet_pretrained = model_cfg.get('resnet_pretrained', False)
    resnet_small_input = model_cfg.get('resnet_small_input', True)
    time_dim = model_cfg.get('time_dim', 128)
    num_blocks = model_cfg.get('num_blocks', 3)
    mlp_ratio = model_cfg.get('mlp_ratio', 4)
    dropout = model_cfg.get('dropout', 0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Evaluation on {device}")

    train_loader, test_loader, flat_dim, num_classes = get_ablation_dataloaders(dataset, batch_size=batch_size, num_workers=num_workers)

    if adapter_type == 'conv':
        (vA, _), _ = train_loader.dataset[0]
        adapter = ConvAdapter(tuple(vA.shape), embed_dim=embed_dim, patch_size=patch_size).to(device)
    elif adapter_type == 'lightconv':
        (vA, _), _ = train_loader.dataset[0]
        adapter = LightConvAdapter(tuple(vA.shape), embed_dim=embed_dim).to(device)
    elif adapter_type == 'resnet':
        (vA, _), _ = train_loader.dataset[0]
        adapter = ResNetAdapter(tuple(vA.shape), embed_dim=embed_dim, arch=resnet_arch,
                                pretrained=resnet_pretrained, small_input=resnet_small_input).to(device)
    else:
        adapter = Adapter(flat_dim, embed_dim=embed_dim, hidden_dim=adapter_hidden_dim).to(device)
    encoder = RecurrentEncoder(num_features=embed_dim, time_dim=time_dim, num_blocks=num_blocks,
                               dropout=dropout, delta=delta, mlp_ratio=mlp_ratio).to(device)
    decoder = BidirectionalDecoder(num_features=embed_dim, time_dim=time_dim, num_blocks=num_blocks,
                                   dropout=dropout, delta=delta, mlp_ratio=mlp_ratio).to(device)

    ckpt_dir = Path(f"checkpoints/grim/{run_name}")

    adapter.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location=device))
    encoder.load_state_dict(torch.load(ckpt_dir / f"encoder_t{T-1}.pt", map_location=device))
    decoder.load_state_dict(torch.load(ckpt_dir / f"decoder_t{T-1}.pt", map_location=device))

    run_evaluation(adapter, encoder, decoder, test_loader, train_loader, T, device, eval_config['linear_probe'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grim/train.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/grim/eval.yaml")
    args = parser.parse_args()
    eval_grim(args.config, args.eval_config)
