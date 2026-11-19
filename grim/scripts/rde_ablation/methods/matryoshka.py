import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.rde_ablation.ablation_model import BaseAblationRunner
from src.loss import matryoshka_info_nce_loss

class MatryoshkaRunner(BaseAblationRunner):
    def compute_loss(self, z1a, z1b, z2a, z2b):
        matryoshka_kwargs = self.config.get('matryoshka', {})
        temperature = self.config['training']['temperature']
        sign_bias = self.config['training'].get('sign_bias', 0.0)

        loss1 = matryoshka_info_nce_loss(z1a, z1b, **matryoshka_kwargs, temperature=temperature, sign_bias=sign_bias)
        loss2 = matryoshka_info_nce_loss(z2a, z2b, **matryoshka_kwargs, temperature=temperature, sign_bias=sign_bias)

        return loss1, loss2

if __name__ == '__main__':
    MatryoshkaRunner().run(default_config_path='configs/ablations/matryoshka.yaml')