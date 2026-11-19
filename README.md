# Locality and Redundancy

Code for the technical report *Locality and Redundancy: Attempts at Learning Without End-to-End Back-Propagation*
(Federico Rubbi, 2026). Each folder is a separate project; run its commands from inside the folder after
`pip install -r requirements.txt`.

| Folder | Section | Run |
|---|---|---|
| `itmf/` | 4.1 IT-MF | `python itmf_mlp.py --dataset CIFAR10`, `python bp_mlp.py --dataset CIFAR10`; reported runs: `bash run_rerun.sh` |
| `vit/` | 4.2 Greedy ViT, InfoViT | `python prepare_data.py`, then `python train_greedy_vit.py`, `train_infovit.py`, `train_bp_vit.py` |
| `dcsm/` | 4.3 DCSM | `bash scripts/run_pretrain.sh`, `bash scripts/train_dcsm.sh configs/train_dcsm_supcon.yaml`, `bash scripts/eval_dcsm.sh <run_dir>` |
| `grim/` | 4.4 GRIM, SRR | `bash scripts/rde_ablation/run_ablation.sh all`, `python scripts/grim/train_grim.py --config configs/grim/train_lightconv.yaml`, `bash scripts/grim/run_srr.sh configs/grim/train_srr_v0.yaml` |
| `fedsym/` | 4.5 FedSym | `python train.py --help`, `python scripts/launch.py <experiment>` |
| `paper/` | all | `bash scripts/figures_all.sh`: figures and tables of the report from the results in `data/` |

MIT license.
