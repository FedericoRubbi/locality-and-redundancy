import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import torch
from scripts.plotting import build_and_log_plot_tables
from src.metrics import compute_dimensional_entropy, compute_cse_and_auc, compute_subspace_metrics, compute_effective_rank, run_linear_probe


def _get_eval_config(config):
    evaluation = config.get('evaluation', {}) if isinstance(config, dict) else {}
    return evaluation or {}

def _subsample_tensors(*tensors, max_samples=None, generator=None):
    if max_samples is None:
        return tensors

    first = tensors[0]
    if first.shape[0] <= max_samples:
        return tensors

    device = first.device
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)

    indices = torch.randperm(first.shape[0], device=device, generator=generator)[:max_samples]
    return tuple(tensor.index_select(0, indices) for tensor in tensors)

def run_evaluation_suite(encoder1, encoder2, loader, config, accelerator, collect_embeddings_joint, run_name="default_run"):
    train_loader, test_loader = loader
    eval_config = _get_eval_config(config)
    timing_enabled = bool(eval_config.get('timing', True))
    max_eval_samples = eval_config.get('max_eval_samples', None)
    probe_max_samples = eval_config.get('probe_max_samples', max_eval_samples)
    probe_epochs = int(eval_config.get('probe_epochs', 25))
    probe_lr = float(eval_config.get('probe_lr', 1e-2))
    probe_batch_size = int(eval_config.get('probe_batch_size', 256))
    uniformity_max_samples = int(eval_config.get('uniformity_max_samples', max_eval_samples if max_eval_samples is not None else 8192))
    uniformity_block_size = int(eval_config.get('uniformity_block_size', 2048))
    uniformity_t = float(eval_config.get('uniformity_t', 2.0))
    effective_rank_max_samples = eval_config.get('effective_rank_max_samples', max_eval_samples)
    
    now = time.perf_counter

    t0 = now()
    X_train_A, X_train_B, y_train = collect_embeddings_joint(train_loader, encoder1, encoder2)
    X_test_A, X_test_B, y_test = collect_embeddings_joint(test_loader, encoder1, encoder2)
    if timing_enabled:
        print(f"[EVAL TIMING] Embedding Collection: {now() - t0:.2f}s")

    if not accelerator.is_main_process:
        return {}

    X_train_A, X_train_B, y_train = _subsample_tensors(X_train_A, X_train_B, y_train, max_samples=max_eval_samples)
    X_test_A, X_test_B, y_test = _subsample_tensors(X_test_A, X_test_B, y_test, max_samples=max_eval_samples)

    metrics = {}
    embedding_dim = X_train_A.shape[1]
    subspace_dims = {
        f"1. (1/1) {embedding_dim}d": embedding_dim,
        f"2. (1/2) {embedding_dim // 2}d": embedding_dim // 2,
        f"3. (1/4) {embedding_dim // 4}d": embedding_dim // 4,
        f"4. (1/8) {embedding_dim // 8}d": embedding_dim // 8,
        f"5. (1/16) {embedding_dim // 16}d": embedding_dim // 16,
    }
    
    t_start = now()
    ent_A = compute_dimensional_entropy(X_train_A)
    ent_B = compute_dimensional_entropy(X_train_B)
    metrics['dimensional_entropy/A'] = ent_A
    metrics['dimensional_entropy/B'] = ent_B
    metrics['dimensional_entropy/mean'] = (ent_A + ent_B) / 2.0

    _, auc_A = compute_cse_and_auc(X_train_A)
    _, auc_B = compute_cse_and_auc(X_train_B)
    metrics['cse_auc/A'] = auc_A
    metrics['cse_auc/B'] = auc_B
    metrics['cse_auc/mean'] = (auc_A + auc_B) / 2.0

    if timing_enabled:
        print(f"[EVAL TIMING] Dimensional Energy: {now() - t_start:.2f}s")

    t_start = now()
    X_train_A_probe, y_train_probe = _subsample_tensors(X_train_A, y_train, max_samples=probe_max_samples)
    X_test_A_probe, y_test_probe = _subsample_tensors(X_test_A, y_test, max_samples=probe_max_samples)

    for name, dim in subspace_dims.items():
        if dim > 0:
            acc = run_linear_probe(
                X_train_A_probe[:, :dim], y_train_probe,
                X_test_A_probe[:, :dim], y_test_probe,
                epochs=probe_epochs, lr=probe_lr, batch_size=probe_batch_size,
            )
            metrics[f'lp_acc_A/{name}'] = acc
     
    if timing_enabled:
        print(f"[EVAL TIMING] Subspace Probing (Linear Probe): {now() - t_start:.2f}s")

    t_start = now()
    X_train_A_unif, X_train_B_unif = _subsample_tensors(X_train_A, X_train_B, max_samples=uniformity_max_samples)
    for name, dim in subspace_dims.items():
        if dim > 0:
            align, unif = compute_subspace_metrics(X_train_A_unif, X_train_B_unif, dim, t=uniformity_t, block_size=uniformity_block_size)
            metrics[f'alignment/{name}'] = align
            metrics[f'uniformity/{name}'] = unif
        
    if timing_enabled:
        print(f"[EVAL TIMING] Alignment & Uniformity: {now() - t_start:.2f}s")

    t_start = now()
    X_train_A_rank, X_train_B_rank = _subsample_tensors(X_train_A, X_train_B, max_samples=effective_rank_max_samples)
    d_eff_A = compute_effective_rank(X_train_A_rank)
    d_eff_B = compute_effective_rank(X_train_B_rank)
    
    if d_eff_A is not None:
        metrics['effective_rank/A'] = d_eff_A
    if d_eff_B is not None:
        metrics['effective_rank/B'] = d_eff_B
    if d_eff_A is not None and d_eff_B is not None:
        metrics['effective_rank/mean'] = (d_eff_A + d_eff_B) / 2.0
        
    if timing_enabled:
        print(f"[EVAL TIMING] Effective Rank SVD: {now() - t_start:.2f}s")

    build_and_log_plot_tables(metrics, run_name, subspace_dims)

    return metrics