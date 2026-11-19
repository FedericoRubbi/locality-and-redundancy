import os
import sys
import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

torch.set_float32_matmul_precision('high')

from data.data import get_ablation_dataloaders, get_probe_train_loader
from src.srr_models import SRRModel
from src.metrics import run_linear_probe


def compute_srr_trajectories(model, loader, T, device, keep_depths, desc,
                             feature="h", amp=True):
    keep = set(keep_depths)
    want_features = feature == "f"
    reprs: dict[int, list[torch.Tensor]] = {t: [] for t in keep}
    all_labels: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc):
            v = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            v = v.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
                outs = model.rollout(v, T, return_features=want_features)
            for t in keep:
                o = outs[t][0] if want_features else outs[t]
                reprs[t].append(o.float().cpu())
            all_labels.append(labels)

    return ({t: torch.cat(chunks, dim=0) for t, chunks in reprs.items()},
            torch.cat(all_labels, dim=0))


def subsample_probe_set(X, y, max_samples):
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    idx = torch.randperm(X.shape[0])[:max_samples]
    return X[idx], y[idx]


def probe_depths(model, train_loader, test_loader, T, device, linear_probe_config,
                 depths, feature="h", amp=True):
    max_samples = linear_probe_config.get('max_samples')
    train_reprs, train_labels = compute_srr_trajectories(
        model, train_loader, T, device, depths, "Probe-train trajectories", feature, amp)
    test_reprs, test_labels = compute_srr_trajectories(
        model, test_loader, T, device, depths, "Probe-test trajectories", feature, amp)

    accuracies = {}
    for t in sorted(depths):
        X_train, y_train = subsample_probe_set(train_reprs[t], train_labels, max_samples)
        accuracies[t] = run_linear_probe(
            X_train, y_train, test_reprs[t], test_labels,
            epochs=linear_probe_config['epochs'],
            lr=linear_probe_config['lr'],
            batch_size=linear_probe_config['batch_size'])
        print(f"  depth {t} ({feature}) | accuracy: {accuracies[t] * 100:.2f}%")
    return accuracies


def run_full_evaluation(model, train_loader, test_loader, T, device, linear_probe_config,
                        feature="h", amp=True):
    max_samples = linear_probe_config.get('max_samples')
    all_depths = list(range(T + 1))

    print(f"\n--- Full evaluation over depths 0..{T} (readout: {feature}) ---")
    train_reprs, train_labels = compute_srr_trajectories(
        model, train_loader, T, device, all_depths, "Train trajectories", feature, amp)
    test_reprs, test_labels = compute_srr_trajectories(
        model, test_loader, T, device, all_depths, "Test trajectories", feature, amp)

    accuracy_vs_depth = []
    for t in all_depths:
        X_train, y_train = subsample_probe_set(train_reprs[t], train_labels, max_samples)
        acc = run_linear_probe(
            X_train, y_train, test_reprs[t], test_labels,
            epochs=linear_probe_config['epochs'],
            lr=linear_probe_config['lr'],
            batch_size=linear_probe_config['batch_size'])
        print(f"  t={t} | accuracy: {acc * 100:.2f}%")
        accuracy_vs_depth.append(acc)

    print("\n--- Elastic probe (head trained at max depth) ---")
    X_train, y_train = subsample_probe_set(train_reprs[T], train_labels, max_samples)
    deep_probe = run_linear_probe(
        X_train, y_train, test_reprs[T], test_labels,
        epochs=linear_probe_config['epochs'],
        lr=linear_probe_config['lr'],
        batch_size=linear_probe_config['batch_size'],
        return_model=True)

    elastic_accuracy_vs_depth = []
    for t in all_depths:
        acc = run_linear_probe(
            X_train, y_train, test_reprs[t], test_labels, model=deep_probe,
            epochs=linear_probe_config['epochs'],
            lr=linear_probe_config['lr'],
            batch_size=linear_probe_config['batch_size'])
        print(f"  elastic t={t}: {acc * 100:.2f}%")
        elastic_accuracy_vs_depth.append(acc)

    return {
        "depths": all_depths,
        "accuracy_vs_depth": accuracy_vs_depth,
        "elastic_accuracy_vs_depth": elastic_accuracy_vs_depth,
    }


def build_model(config, in_shape) -> SRRModel:
    m = config['model']
    return SRRModel(
        in_shape,
        channels=m.get('channels', 256),
        stem_strides=tuple(m.get('stem_strides', [2, 2])),
        num_blocks=m.get('num_blocks', 3),
        time_dim=m.get('time_dim', 128),
        embed_dim=m.get('embed_dim', 1024),
        pool=m.get('pool', 2),
        delta_task=float(m.get('delta_task', 0.5)),
        alpha=float(m.get('alpha', 0.5)),
        dropout=float(m.get('dropout', 0.0)),
        tangent_projection=m.get('tangent_projection', True),
        dense_head=m.get('dense_head', False),
        dense_embed_dim=m.get('dense_embed_dim', 256),
        masked_predictor=m.get('masked_predictor', False),
    )


def eval_srr(config_path: str, eval_config_path: str, checkpoint_path: str | None = None) -> None:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(eval_config_path, 'r') as f:
        eval_config = yaml.safe_load(f)

    run_name = config['run_name']
    dataset = config['training']['dataset']
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 8)
    T = config['training']['T']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, _, _ = get_ablation_dataloaders(
        dataset, batch_size=batch_size, num_workers=num_workers)
    probe_train_loader = get_probe_train_loader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    (vA, _), _ = train_loader.dataset[0]
    model = build_model(config, tuple(vA.shape)).to(device)
    ckpt = Path(checkpoint_path) if checkpoint_path else Path(f"checkpoints/grim/{run_name}") / "srr_model.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    print(f"Loaded {ckpt}")

    for feature in ("h", "f"):
        run_full_evaluation(model, probe_train_loader, test_loader, T, device,
                            eval_config['linear_probe'], feature=feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grim/train_srr_v0.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/grim/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="override checkpoint path, e.g. an *_interrupted.pt snapshot")
    args = parser.parse_args()
    eval_srr(args.config, args.eval_config, args.checkpoint)
