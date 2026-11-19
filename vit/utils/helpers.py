import os
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(save_dir):
    log_file = os.path.join(save_dir, 'log.txt')
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    return log_file

def get_incremental_save_dir(base_dir, dataset_name):
    dataset_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    run_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith('run_')]
    
    latest_run = 0
    for d in run_dirs:
        try:
            run_num = int(d.split('_')[1])
            if run_num > latest_run:
                latest_run = run_num
        except ValueError:
            continue
            
    new_run_num = latest_run + 1
    new_run_dir = os.path.join(dataset_dir, f'run_{new_run_num:03d}')
    os.makedirs(new_run_dir, exist_ok=True)
    
    return new_run_dir
