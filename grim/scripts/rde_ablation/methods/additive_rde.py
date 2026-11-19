import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.rde_ablation.ablation_model import BaseAblationRunner
from src.rde import get_variance_schedule, AsymmetricAdditiveNoise
from src.loss import compensated_asymmetric_info_nce_loss, asymmetric_info_nce_loss
from src.metrics import compute_representational_drift

class AdditiveNoiseRunner(BaseAblationRunner):
    def setup_modules(self, input_dim):
        self.schedule = get_variance_schedule(dim=self.config['training']['embed_dim'], **self.config['rde'])
        self.rde_injector1 = AsymmetricAdditiveNoise(self.schedule).to(self.accelerator.device)
        self.rde_injector2 = AsymmetricAdditiveNoise(self.schedule).to(self.accelerator.device)

    def compute_loss(self, z1a, z1b, z2a, z2b):
        z1a_noisy = self.rde_injector1(z1a)
        z2a_noisy = self.rde_injector2(z2a)
        local_schedule = self.schedule.to(self.accelerator.device)
        temperature = self.config['training']['temperature']
        sign_bias = self.config['training']['sign_bias']
        moment_bias = self.config['training']['moment_bias']

        if not moment_bias:
            loss1 = compensated_asymmetric_info_nce_loss(z1a_noisy, z1b, local_schedule, mode="additive", temperature=temperature)
            loss2 = compensated_asymmetric_info_nce_loss(z2a_noisy, z2b, local_schedule, mode="additive", temperature=temperature)
        else:
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
    AdditiveNoiseRunner().run(default_config_path='configs/ablations/additive_rde.yaml')