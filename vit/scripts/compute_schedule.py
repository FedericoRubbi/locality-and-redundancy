import argparse
import os
import torch
import json
import matplotlib.pyplot as plt
from utils.info_schedule import compute_schedule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=100)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--mc-samples', type=int, default=5000)
    parser.add_argument('--save-dir', type=str, default='configs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computing capacity schedule using {device} with {args.mc_samples} MC samples.")
    
    s_values, capacities = compute_schedule(K=args.classes, L=args.layers, N=args.mc_samples, device=device)
    
    output_path = os.path.join(args.save_dir, 'schedule.json')
    with open(output_path, 'w') as f:
        json.dump({'s_values': s_values, 'capacities': capacities}, f)
        
    print(f"Saved schedule to {output_path}")
    
    plt.figure()
    plt.plot(range(1, args.layers + 1), capacities, marker='o')
    plt.title('Computed Capacity Schedule')
    plt.xlabel('Layer')
    plt.ylabel('Capacity (nats)')
    plt.savefig(os.path.join(args.save_dir, 'capacity_schedule.png'))
    plt.close()
    
if __name__ == '__main__':
    main()
