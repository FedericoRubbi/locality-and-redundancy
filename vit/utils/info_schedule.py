import torch
import numpy as np

def monte_carlo_capacity(s, c, K, N=10000, device='cpu'):
    y_n = torch.randint(0, K, (N,), device=device)
    
    alpha = torch.full((N, K), c, device=device)
    alpha[torch.arange(N), y_n] += s
    
    dirichlet = torch.distributions.Dirichlet(alpha)
    y_tilde = dirichlet.sample()
    
    logits = s * torch.log(y_tilde + 1e-10)
    p = torch.nn.functional.softmax(logits, dim=-1)
    
    log_p = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -torch.sum(p * log_p, dim=-1)
    
    expected_entropy = entropy.mean().item()
    
    capacity = np.log(K) - expected_entropy
    return capacity

def compute_schedule(K, L, c=1.0, epsilon=0.05, N=10000, max_s=100.0, tol=1e-3, device='cpu'):
    s_schedule = []
    capacities = []
    
    for l in range(1, L + 1):
        rho_l = epsilon + (1 - 2 * epsilon) * ((l - 1) / max(1, L - 1))
        target_I = rho_l * np.log(K)
        
        low = 0.0
        high = max_s
        s_l = 0.0
        
        for _ in range(50):
            mid = (low + high) / 2
            cap = monte_carlo_capacity(mid, c, K, N, device=device)
            
            if abs(cap - target_I) < tol:
                s_l = mid
                break
            elif cap < target_I:
                low = mid
            else:
                high = mid
                
        s_l = (low + high) / 2
        final_cap = monte_carlo_capacity(s_l, c, K, N, device=device)
        
        s_schedule.append(s_l)
        capacities.append(final_cap)
        print(f"Layer {l}/{L}: target_I={target_I:.4f}, cap={final_cap:.4f}, s_l={s_l:.4f}")
        
    return s_schedule, capacities
