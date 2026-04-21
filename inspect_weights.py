import torch

with open("keys.txt", "w") as f:
    b = torch.load('baseline_model.pth', map_location='cpu', weights_only=True)
    h = torch.load('hybrid_model.pth', map_location='cpu', weights_only=True)
    
    f.write("Baseline Keys & Shapes:\n")
    for k, v in b.items():
        f.write(f"  {k}: {v.shape}\n")
        
    f.write("\nHybrid Keys & Shapes:\n")
    for k, v in h.items():
        f.write(f"  {k}: {v.shape}\n")
