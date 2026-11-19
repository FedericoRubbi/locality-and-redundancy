import os
import shutil
import kagglehub

def prepare_dataset():
    target_dir = "./data/imagenet100"
    
    print("Downloading ImageNet-100 via KaggleHub (approx. 17GB, this may take a while)...")
    cache_path = kagglehub.dataset_download("ambityga/imagenet100")
    
    print(f"Downloaded to cache: {cache_path}")
    print(f"Copying files to target directory: {target_dir}...")
    
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)
    
    
    if os.path.exists(os.path.join(target_dir, "train.X1")):
        print("Found split training/validation directories from ambityga dataset. Fixing...")
        train_path = os.path.join(target_dir, "train")
        val_path = os.path.join(target_dir, "val")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        
        for suffix in ["X1", "X2", "X3", "X4"]:
            src = os.path.join(target_dir, f"train.{suffix}")
            if os.path.exists(src):
                for item in os.listdir(src):
                    dest = os.path.join(train_path, item)
                    if not os.path.exists(dest):
                        shutil.move(os.path.join(src, item), train_path)
                if not os.listdir(src):
                    os.rmdir(src)
                else:
                    shutil.rmtree(src, ignore_errors=True)
                
        val_src = os.path.join(target_dir, "val.X")
        if os.path.exists(val_src):
            for item in os.listdir(val_src):
                dest = os.path.join(val_path, item)
                if not os.path.exists(dest):
                    shutil.move(os.path.join(val_src, item), val_path)
            if not os.listdir(val_src):
                os.rmdir(val_src)
            else:
                shutil.rmtree(val_src, ignore_errors=True)
    
    train_path_check = os.path.join(target_dir, "train")
    
    if not os.path.exists(train_path_check):
        for root, dirs, files in os.walk(target_dir):
            if "train" in dirs:
                nested_root = root
                print(f"Found nested dataset at {nested_root}. Flattening structure...")
                for item in os.listdir(nested_root):
                    dest = os.path.join(target_dir, item)
                    if not os.path.exists(dest):
                        shutil.move(os.path.join(nested_root, item), target_dir)
                break
                
    if os.path.exists(os.path.join(target_dir, "train")) and os.path.exists(os.path.join(target_dir, "val")):
        print(f"Success! The dataset is properly formatted at {target_dir}")
        print("You can now safely run: python3 train.py")
    else:
        print(f"Warning: Could not locate 'train' and 'val' folders in {target_dir}. Please check the directory manually.")

if __name__ == "__main__":
    prepare_dataset()
    prepare_dataset()