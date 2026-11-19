import torch

def generate_dirichlet_targets(y, s_l, c_l, K):
    batch_size = y.size(0)
    device = y.device
    
    alpha = torch.full((batch_size, K), c_l, device=device)
    alpha[torch.arange(batch_size), y] += s_l
    
    dirichlet = torch.distributions.Dirichlet(alpha)
    y_tilde = dirichlet.sample()
    return y_tilde
