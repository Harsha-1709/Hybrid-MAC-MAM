import torch
import torch.nn.functional as F
import time

def original_mam(x, weight, kernel_size, padding, stride, out_channels):
    batch_size, in_channels, h_in, w_in = x.size()
    x_unfolded = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
    L = x_unfolded.size(-1)
    
    x_unfolded = x_unfolded.view(batch_size, 1, in_channels, kernel_size, kernel_size, L)
    w_expanded = weight.unsqueeze(0).unsqueeze(-1)
    wx = x_unfolded * w_expanded
    wx_flat = wx.view(batch_size, out_channels, -1, L)
    y_mam, _ = torch.max(wx_flat, dim=2)
    return y_mam

def optimized_mam(x, weight, kernel_size, padding, stride, out_channels):
    batch_size, in_channels, h_in, w_in = x.size()
    x_unfolded = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
    L = x_unfolded.size(-1)
    
    x_unfolded = x_unfolded.view(batch_size, 1, -1, L)
    w_flat = weight.view(out_channels, -1).unsqueeze(0).unsqueeze(-1)
    
    y_mam_list = []
    for o in range(out_channels):
        w_o = w_flat[:, o:o+1, :, :]
        # [batch_size, 1, C*K*K, L] -> max dim=2 -> [batch_size, 1, L]
        out_o = torch.max(x_unfolded * w_o, dim=2)[0]
        y_mam_list.append(out_o)
        
    y_mam = torch.cat(y_mam_list, dim=1)
    return y_mam

# Test
B, C, H, W = 16, 32, 16, 16
out_channels = 64
kernel_size = 3
padding = 1
stride = 1

x = torch.randn(B, C, H, W)
weight = torch.randn(out_channels, C, kernel_size, kernel_size)

# Correctness
y1 = original_mam(x, weight, kernel_size, padding, stride, out_channels)
y2 = optimized_mam(x, weight, kernel_size, padding, stride, out_channels)

print("Max difference:", torch.max(torch.abs(y1 - y2)).item())

# Speed
t0 = time.time()
for _ in range(10):
    original_mam(x, weight, kernel_size, padding, stride, out_channels)
t1 = time.time()
print(f"Original Time: {t1 - t0:.4f}s")

t0 = time.time()
for _ in range(10):
    optimized_mam(x, weight, kernel_size, padding, stride, out_channels)
t1 = time.time()
print(f"Optimized Time: {t1 - t0:.4f}s")
