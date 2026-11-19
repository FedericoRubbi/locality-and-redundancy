import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.rde_ablation.ablation_model import BaseAblationRunner
from src.rde import CumulativeMultiplicativeBetaNoise
from src.loss import asymmetric_info_nce_loss
from src.metrics import compute_representational_drift

class CumulativeMultiplicativeNoiseRunner(BaseAblationRunner):
    def setup_modules(self, input_dim):
        rde_config = self.config.get('rde', {})
        mu = rde_config.get('mu', 0.98)
        nu = rde_config.get('nu', 100.0)
        renormalize = rde_config.get('renormalize', True)
        
        embed_dim = self.config['training']['embed_dim']
        
        self.rde_injector1 = CumulativeMultiplicativeBetaNoise(
            dim=embed_dim, mu=mu, nu=nu, renormalize=renormalize
        ).to(self.accelerator.device)
        
        self.rde_injector2 = CumulativeMultiplicativeBetaNoise(
            dim=embed_dim, mu=mu, nu=nu, renormalize=renormalize
        ).to(self.accelerator.device)

    def compute_loss(self, z1a, z1b, z2a, z2b):
        z1a_noisy, z1b_noisy = self.rde_injector1(z1a), self.rde_injector1(z1b)
        z2a_noisy, z2b_noisy = self.rde_injector2(z2a), self.rde_injector2(z2b)
        
        temperature = self.config['training']['temperature']
        sign_bias = self.config['training']['sign_bias']
        
        loss1 = asymmetric_info_nce_loss(z1a_noisy, z1b_noisy, temperature=temperature, gamma=sign_bias)
        loss2 = asymmetric_info_nce_loss(z2a_noisy, z2b_noisy, temperature=temperature, gamma=sign_bias)
        
        return loss1, loss2

    def post_eval_hook(self, eval_train_loader, metrics):
        train_embs1, _, _ = self._collect_embeddings_joint(eval_train_loader, self.enc1, self.enc2)
        if self.accelerator.is_main_process:
            train_embs1_noisy = self.rde_injector1(train_embs1.to(self.accelerator.device)).cpu()
            metrics['representational_drift'] = compute_representational_drift(train_embs1, train_embs1_noisy)
        return metrics

if __name__ == '__main__':
    CumulativeMultiplicativeNoiseRunner().run(
        default_config_path='configs/ablations/cumulative_multiplicative_rde.yaml'
    )