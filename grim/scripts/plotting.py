import wandb

def build_and_log_plot_tables(metrics, run_name, subspace_dims):
    lp_table = wandb.Table(columns=["method", "subspace", "value"])
    alignment_table = wandb.Table(columns=["method", "subspace", "value"])
    uniformity_table = wandb.Table(columns=["method", "subspace", "value"])

    for label in subspace_dims.keys():
        if f"lp_acc_A/{label}" in metrics:
            lp_table.add_data(run_name, label, metrics[f"lp_acc_A/{label}"])
        if f"alignment/{label}" in metrics:
            alignment_table.add_data(run_name, label, metrics[f"alignment/{label}"])
        if f"uniformity/{label}" in metrics:
            uniformity_table.add_data(run_name, label, metrics[f"uniformity/{label}"])

    entropy_table = wandb.Table(columns=["method", "target", "value"])
    cse_table = wandb.Table(columns=["method", "target", "value"])
    rank_table = wandb.Table(columns=["method", "target", "value"])

    global_mappings = {
        'dimensional_entropy': entropy_table,
        'cse_auc': cse_table,
        'effective_rank': rank_table
    }

    for base_key, table in global_mappings.items():
        for target in ['A', 'B', 'mean']:
            dict_key = f"{base_key}/{target}"
            if dict_key in metrics:
                clean_target = "Mean" if target == 'mean' else target
                table.add_data(run_name, clean_target, metrics[dict_key])

    wandb.log({
        "plot_data/linear_probe_accuracy": lp_table,
        "plot_data/subspace_alignment": alignment_table,
        "plot_data/subspace_uniformity": uniformity_table,
        "plot_data/dimensional_entropy": entropy_table,
        "plot_data/cse_auc": cse_table,
        "plot_data/effective_rank": rank_table
    })