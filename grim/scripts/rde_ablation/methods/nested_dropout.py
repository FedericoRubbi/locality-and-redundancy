import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.rde_ablation.ablation_model import BaseAblationRunner
from src.rde import DeterministicNestedDropout
from src.loss import asymmetric_info_nce_loss
from src.metrics import compute_representational_drift

class NestedDropoutRunner(BaseAblationRunner):
    def setup_modules(self, input_dim):
        embed_dim = self.config['training']['embed_dim']
        dropout_opts = self.config.get('dropout', {})
        self.nested_dropout1 = DeterministicNestedDropout(dim=embed_dim, **dropout_opts)
        self.nested_dropout2 = DeterministicNestedDropout(dim=embed_dim, **dropout_opts)

    def compute_loss(self, z1a, z1b, z2a, z2b):
        z1a_dropped, z1b_dropped = self.nested_dropout1(z1a), self.nested_dropout1(z1b)
        z2a_dropped, z2b_dropped = self.nested_dropout2(z2a), self.nested_dropout2(z2b)

        temperature = self.config['training']['temperature']
        sign_bias = self.config['training']['sign_bias']
        
        loss1 = asymmetric_info_nce_loss(z1a_dropped, z1b_dropped, temperature=temperature, gamma=sign_bias)
        loss2 = asymmetric_info_nce_loss(z2a_dropped, z2b_dropped, temperature=temperature, gamma=sign_bias)
        
        return loss1, loss2

    def post_eval_hook(self, eval_train_loader, metrics):
        train_embs1, _, _ = self._collect_embeddings_joint(eval_train_loader, self.enc1, self.enc2)
        if self.accelerator.is_main_process:
            train_embs1_dropped = self.nested_dropout1(train_embs1.to(self.accelerator.device)).cpu()
            metrics['representational_drift'] = compute_representational_drift(train_embs1, train_embs1_dropped)
        return metrics

if __name__ == '__main__':
    NestedDropoutRunner().run(default_config_path='configs/ablations/nested_dropout.yaml')