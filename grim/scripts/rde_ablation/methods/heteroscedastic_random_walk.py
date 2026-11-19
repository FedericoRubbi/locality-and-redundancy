import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.rde_ablation.ablation_model import BaseAblationRunner
from src.rde import HeteroscedasticRandomWalkNoise
from src.loss import asymmetric_info_nce_loss
from src.metrics import compute_representational_drift


class HeteroscedasticRandomWalkRunner(BaseAblationRunner):
    def setup_modules(self, input_dim):
        rde_config = self.config.get('rde', {})
        gamma = rde_config.get('gamma', 1.0)
        v_max = rde_config.get('v_max', 0.005)
        renormalize = rde_config.get('renormalize', True)

        embed_dim = self.config['training']['embed_dim']

        self.rde_injector1 = HeteroscedasticRandomWalkNoise(
            dim=embed_dim,
            gamma=gamma,
            v_max=v_max,
            renormalize=renormalize,
        ).to(self.accelerator.device)

        self.rde_injector2 = HeteroscedasticRandomWalkNoise(
            dim=embed_dim,
            gamma=gamma,
            v_max=v_max,
            renormalize=renormalize,
        ).to(self.accelerator.device)

    def compute_loss(self, z1a, z1b, z2a, z2b):
        temperature = self.config['training']['temperature']
        sign_bias = self.config['training']['sign_bias']

        z1a_noisy = self.rde_injector1(z1a)
        z2a_noisy = self.rde_injector2(z2a)

        loss1 = asymmetric_info_nce_loss(z1a_noisy, z1b, temperature=temperature, gamma=sign_bias)
        loss2 = asymmetric_info_nce_loss(z2a_noisy, z2b, temperature=temperature, gamma=sign_bias)

        return loss1, loss2

    def post_eval_hook(self, eval_train_loader, metrics):
        train_embs1, _, _ = self._collect_embeddings_joint(eval_train_loader, self.enc1, self.enc2)
        if self.accelerator.is_main_process:
            train_embs1_noisy = self.rde_injector1(train_embs1.to(self.accelerator.device)).cpu()
            metrics['representational_drift'] = compute_representational_drift(train_embs1, train_embs1_noisy)
        return metrics


if __name__ == '__main__':
    HeteroscedasticRandomWalkRunner().run(default_config_path='configs/ablations/heteroscedastic_random_walk.yaml')