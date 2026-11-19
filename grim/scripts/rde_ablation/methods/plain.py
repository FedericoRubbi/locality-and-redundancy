import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.rde_ablation.ablation_model import BaseAblationRunner
from src.loss import asymmetric_info_nce_loss

class PlainRunner(BaseAblationRunner):
    def compute_loss(self, z1a, z1b, z2a, z2b):
        temperature = self.config['training']['temperature']
        sign_bias = self.config['training']['sign_bias']
        loss1 = asymmetric_info_nce_loss(z1a, z1b, temperature=temperature, gamma=sign_bias)
        loss2 = asymmetric_info_nce_loss(z2a, z2b, temperature=temperature, gamma=sign_bias)
        return loss1, loss2

if __name__ == '__main__':
    PlainRunner().run(default_config_path='configs/ablations/plain.yaml')