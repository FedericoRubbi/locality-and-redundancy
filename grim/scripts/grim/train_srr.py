import os
import sys
import copy
import math
import random
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
import yaml
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

torch.set_float32_matmul_precision('high')

from data.data import get_ablation_dataloaders, get_probe_train_loader
from src.objectives import build_objective, StepContext
from src.pcgrad import flatten_grads, assign_flat_grad, gradient_stats, pcgrad_causal
from scripts.grim.eval_srr import build_model, probe_depths, run_full_evaluation


class SRRTrainer:
    def __init__(self, config_path: str, eval_config_path: str, resume: bool = False):
        self.init_config(config_path, eval_config_path)
        self.init_data()
        self.init_model()
        self.resume_state = self.load_train_state() if resume else None
        self.init_wandb()
        print(f"Running on {self.device} | regime {self.regime}")

    def init_config(self, config_path: str, eval_config_path: str) -> None:
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(eval_config_path, 'r') as f:
            self.eval_config = yaml.safe_load(f)

        t = self.config['training']
        self.run_name = self.config['run_name']
        self.dataset = t['dataset']
        self.num_workers = t.get('num_workers', 8)
        self.batch_size = t['batch_size']
        self.epochs = t['epochs']
        self.lr = float(t['lr'])
        self.final_lr = float(t.get('final_lr', 0.0))
        self.weight_decay = float(t.get('weight_decay', 1e-6))
        self.warmup_epochs = t.get('warmup_epochs', 10)
        self.T = t['T']
        self.temperature = float(t['temperature'])
        self.detach_state = t.get('detach_state', False)
        self.loss_at_zero = t.get('loss_at_zero', False)
        self.loss_steps = t.get('loss_steps', 'all')
        assert self.loss_steps in ('all', 'last', 'random'), f"unknown loss_steps: {self.loss_steps}"
        self._step_pool = list(range(0 if self.loss_at_zero else 1, self.T + 1))
        self.loss_step_indices = [self.T] if self.loss_steps == 'last' else self._step_pool

        self.objective_name = t.get('objective', 'global_infonce')
        self.use_ema = t.get('use_ema', False)
        self.ema_momentum = float(t.get('ema_momentum', 0.996))
        self.use_pcgrad = t.get('use_pcgrad', False)
        self.grad_log_interval = t.get('grad_log_interval', 50)
        self.eval_every = t.get('eval_every', 20)
        self.amp = t.get('amp', True)

        self.regime = {
            (False, False): "V0_plain", (True, False): "V1_ema",
            (False, True): "V2_pcgrad", (True, True): "V3_ema_pcgrad",
        }[(self.use_ema, self.use_pcgrad)]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_dir = Path(f"checkpoints/grim/{self.run_name}")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.ckpt_dir / "train_state.pt"

    def init_data(self) -> None:
        self.train_loader, self.test_loader, _, self.num_classes = get_ablation_dataloaders(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.probe_train_loader = get_probe_train_loader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def init_model(self) -> None:
        (vA, _), _ = self.train_loader.dataset[0]
        self.in_shape = tuple(vA.shape)
        self.model = build_model(self.config, self.in_shape).to(self.device)

        self.target = None
        if self.use_ema:
            self.target = copy.deepcopy(self.model).eval()
            for p in self.target.parameters():
                p.requires_grad = False

        grid = self.model.stem.out_shape[-1]
        self.objective = build_objective(self.config['training'], self.T, grid, self.use_ema)

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(self.params, lr=self.lr, weight_decay=self.weight_decay)
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * self.epochs
        self.global_step = 0

        n_params = sum(p.numel() for p in self.params)
        state_shape = self.model.stem.out_shape
        print(f"SRR model: {n_params/1e6:.2f}M params | state {state_shape} "
              f"| effective depth {self.T} x {len(self.model.encoder.blocks)} blocks")

    def load_train_state(self) -> dict | None:
        if not self.state_path.exists():
            print(f"No train state at {self.state_path}; starting from scratch.")
            return None
        state = torch.load(self.state_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.use_ema and state.get('target') is not None:
            self.target.load_state_dict(state['target'])
        self.global_step = state['global_step']
        print(f"Resumed from epoch {state['epoch']} (step {self.global_step}).")
        return state

    def save_train_state(self, epoch: int, eval_history: dict) -> None:
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'target': self.target.state_dict() if self.use_ema else None,
            'epoch': epoch,
            'global_step': self.global_step,
            'eval_history': eval_history,
            'wandb_id': self.wandb_id,
        }, self.state_path)

    def init_wandb(self) -> None:
        prior = self.resume_state['wandb_id'] if self.resume_state else None
        run = wandb.init(project="grim", name=self.run_name,
                         id=prior, resume="allow" if prior else None,
                         config={**self.config, "regime": self.regime,
                                 "n_params": sum(p.numel() for p in self.params)})
        self.wandb_id = run.id

    def lr_at(self, step: int) -> float:
        warmup = self.warmup_epochs * self.steps_per_epoch
        if warmup > 0 and step < warmup:
            return self.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, self.total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return self.final_lr + 0.5 * (self.lr - self.final_lr) * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def update_target(self) -> None:
        m = self.ema_momentum
        for p_t, p_o in zip(self.target.parameters(), self.model.parameters()):
            p_t.mul_(m).add_(p_o.detach(), alpha=1 - m)
        for b_t, b_o in zip(self.target.buffers(), self.model.buffers()):
            b_t.copy_(b_o)

    @property
    def chance_loss(self) -> float:
        B = self.batch_size
        return math.log(B) if self.use_ema else math.log(2 * B - 1)

    def compute_step_losses(self, vA: torch.Tensor, vB: torch.Tensor):
        step_indices = ([random.choice(self._step_pool)] if self.loss_steps == 'random'
                        else self.loss_step_indices)
        want_maps = self.objective.needs_maps
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=self.amp and self.device.type == 'cuda'):
            roll_A = self.model.rollout(vA, self.T, detach_state=self.detach_state,
                                        collect_diagnostics=True, return_maps=want_maps)
            outs_A, maps_A, diagnostics = roll_A if want_maps else (roll_A[0], None, roll_A[1])
            roll_B = self.model.rollout(vB, self.T, detach_state=self.detach_state,
                                        return_maps=want_maps)
            outs_B, maps_B = roll_B if want_maps else (roll_B, None)

            tgt_A = tgt_B = tgt_map_A = tgt_map_B = None
            if self.use_ema:
                with torch.no_grad():
                    t_roll_A = self.target.rollout(vA, self.T, return_maps=want_maps)
                    t_roll_B = self.target.rollout(vB, self.T, return_maps=want_maps)
                    if want_maps:
                        (tgt_A, tgt_map_A), (tgt_B, tgt_map_B) = t_roll_A, t_roll_B
                    else:
                        tgt_A, tgt_B = t_roll_A, t_roll_B

            losses = []
            for t in step_indices:
                ctx = StepContext(
                    t=t, T=self.T, emb_A=outs_A[t], emb_B=outs_B[t],
                    map_A=maps_A[t] if want_maps else None,
                    map_B=maps_B[t] if want_maps else None,
                    tgt_A=tgt_A[t] if self.use_ema else None,
                    tgt_B=tgt_B[t] if self.use_ema else None,
                    tgt_map_A=tgt_map_A[t] if tgt_map_A is not None else None,
                    tgt_map_B=tgt_map_B[t] if tgt_map_B is not None else None,
                    model=self.model, target_model=self.target)
                losses.append(self.objective(ctx).float())
        return losses, diagnostics, step_indices

    def apply_gradients(self, losses: list[torch.Tensor], want_stats: bool) -> dict:
        scale = 1.0 / len(losses)
        stats: dict = {}

        if not (self.use_pcgrad or want_stats):
            self.optimizer.zero_grad(set_to_none=True)
            (sum(losses) * scale).backward()
            return stats

        per_step = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(loss * scale, self.params,
                                        retain_graph=(i < len(losses) - 1),
                                        allow_unused=True)
            per_step.append(flatten_grads(grads, self.params))

        if want_stats:
            stats = gradient_stats(per_step)

        total = pcgrad_causal(per_step) if self.use_pcgrad else sum(per_step)
        self.optimizer.zero_grad(set_to_none=True)
        assign_flat_grad(self.params, total)
        return stats

    def train(self) -> None:
        eval_history: dict[int, dict[int, float]] = {}
        probe_depth_set = sorted({0, self.T // 2, self.T})
        start_epoch = 0
        if self.resume_state:
            eval_history = self.resume_state['eval_history']
            start_epoch = self.resume_state['epoch']

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [{self.regime}]")
            for (vA, vB), _ in pbar:
                vA = vA.to(self.device, non_blocking=True)
                vB = vB.to(self.device, non_blocking=True)

                lr = self.lr_at(self.global_step)
                for group in self.optimizer.param_groups:
                    group['lr'] = lr

                losses, diagnostics, step_indices = self.compute_step_losses(vA, vB)
                want_stats = self.global_step % self.grad_log_interval == 0
                stats = self.apply_gradients(losses, want_stats)

                torch.nn.utils.clip_grad_norm_(self.params, max_norm=5.0)
                self.optimizer.step()
                if self.use_ema:
                    self.update_target()

                self.global_step += 1
                loss_values = [l.item() for l in losses]
                payload = {
                    "global_step": self.global_step,
                    "epoch": epoch,
                    "lr": lr,
                    "losses/mean": sum(loss_values) / len(loss_values),
                    "losses/chance": self.chance_loss,
                    "losses/first_step": loss_values[0],
                    "losses/last_step": loss_values[-1],
                    "losses/depth_gap": loss_values[-1] - loss_values[0],
                }
                payload.update({f"losses/step_{t}": v
                                for t, v in zip(step_indices, loss_values)})
                payload.update(self.diagnostic_payload(diagnostics))
                payload.update(stats)
                payload.update(getattr(self.objective, 'last_stats', {}) or {})
                wandb.log(payload)
                pbar.set_postfix({'L': payload['losses/mean'], 'lr': f"{lr:.2e}"})

            if (epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1:
                torch.save(self.model.state_dict(), self.ckpt_dir / "srr_model.pt")
                print(f"\n--- Probe sweep at epoch {epoch+1} ---")
                accuracies = probe_depths(
                    self.model, self.probe_train_loader, self.test_loader, self.T,
                    self.device, self.eval_config['linear_probe'], probe_depth_set, amp=self.amp)
                eval_history[epoch + 1] = accuracies
                wandb.log({
                    "epoch": epoch,
                    **{f"probe/depth_{d}": a for d, a in accuracies.items()},
                    "probe/depth_gain": accuracies[self.T] - accuracies[0],
                })

            self.save_train_state(epoch + 1, eval_history)

        self.finalize(eval_history)

    def diagnostic_payload(self, diagnostics: list[dict]) -> dict:
        if not diagnostics:
            return {}
        rel = torch.tensor([d["rel_displacement"] for d in diagnostics])
        norms = torch.tensor([d["state_norm"] for d in diagnostics])
        payload = {
            "state/rel_displacement_mean": rel.mean().item(),
            "state/rel_displacement_min": rel.min().item(),
            "state/rel_displacement_max": rel.max().item(),
            "state/norm_first": norms[0].item(),
            "state/norm_last": norms[-1].item(),
            "state/readout_cos_to_stem": diagnostics[-1]["readout_cos_to_stem"],
        }
        angles = [d["step_angle"] for d in diagnostics if "step_angle" in d]
        if angles:
            t_angles = torch.tensor(angles)
            payload["state/mean_step_angle"] = t_angles.mean().item()
            payload["state/max_step_angle"] = t_angles.max().item()
        return payload

    def finalize(self, eval_history: dict) -> None:
        torch.save(self.model.state_dict(), self.ckpt_dir / "srr_model.pt")
        print("\nTraining complete. Running final full evaluation...")

        for feature in ("h", "f"):
            results = run_full_evaluation(
                self.model, self.probe_train_loader, self.test_loader, self.T,
                self.device, self.eval_config['linear_probe'], feature=feature, amp=self.amp)
            for series in ("accuracy_vs_depth", "elastic_accuracy_vs_depth"):
                wandb.log({
                    f"evaluation/{series}_{feature}": wandb.plot.line_series(
                        xs=[results["depths"]], ys=[results[series]], keys=[self.regime],
                        title=f"{series} ({feature})", xname="depth")
                })
            wandb.summary[f"final/best_accuracy_{feature}"] = max(results["accuracy_vs_depth"])
            wandb.summary[f"final/stem_accuracy_{feature}"] = results["accuracy_vs_depth"][0]
            wandb.summary[f"final/deepest_accuracy_{feature}"] = results["accuracy_vs_depth"][-1]
            wandb.summary[f"final/depth_gain_{feature}"] = (
                results["accuracy_vs_depth"][-1] - results["accuracy_vs_depth"][0])

        if eval_history:
            epochs = sorted(eval_history)
            depths = sorted(eval_history[epochs[0]])
            wandb.log({
                "evaluation/probe_vs_epoch": wandb.plot.line_series(
                    xs=epochs, ys=[[eval_history[e][d] for e in epochs] for d in depths],
                    keys=[f"depth_{d}" for d in depths],
                    title="probe accuracy vs epoch", xname="epoch")
            })

        wandb.finish()
        print(f"Done. Checkpoints at {self.ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grim/train_srr_v0.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/grim/eval.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="continue from checkpoints/grim/<run>/train_state.pt if present")
    args = parser.parse_args()
    SRRTrainer(args.config, args.eval_config, resume=args.resume).train()
