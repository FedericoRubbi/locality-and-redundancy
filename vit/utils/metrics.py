import torch

class KDE_Renyi_Entropy:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, z):
        with torch.no_grad():
            dists = torch.cdist(z, z, p=2)**2
            
            K = torch.exp(-dists / (2 * self.sigma ** 2))
            
            return -torch.log(K.mean() + 1e-8)


def track_block_metrics(z_list, logits_list, targets, loss_l_val, k=100):
    metrics = []
    return metrics