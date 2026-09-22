import os

def process_file(src, dst, old_str, new_str, old_lower, new_lower):
    with open(src, 'r') as f:
        content = f.read()
    
    content = content.replace(old_str, new_str)
    content = content.replace(old_lower, new_lower)
    
    with open(dst, 'w') as f:
        f.write(content)

# Prepare signal scripts already copied and modified by multi_replace? Wait, multi_replace did not replace fd001 in print statements but that's okay.
# Let's cleanly copy everything from train_fd001_conditioned.py
process_file("train_fd001_conditioned.py", "train_fd002_conditioned.py", "FD001", "FD002", "fd001", "fd002")
process_file("train_fd001_conditioned.py", "train_fd004_conditioned.py", "FD001", "FD004", "fd001", "fd004")

# Same for slurm scripts
process_file("slurm/kepin_fd001_conditioned.slurm", "slurm/kepin_fd002_conditioned.slurm", "FD001", "FD002", "fd001", "fd002")
process_file("slurm/kepin_fd001_conditioned.slurm", "slurm/kepin_fd004_conditioned.slurm", "FD001", "FD004", "fd001", "fd004")

print("Created training and SLURM scripts.")
