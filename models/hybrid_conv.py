import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridConv2d(nn.Module):
    """
    Adaptive Hybrid Spatial Layer.
    Combines standard 2D Convolution (MAC) with a Spatial Max-Filtering operation (MAM).
    The combination is controlled by a learnable parameter alpha.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, init_alpha=0.5):
        super(HybridConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Standard convolutional weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Learnable adaptive parameter alpha for balancing Conv (MAC) and MAM
        # Sigmoid will be applied to keep it in [0, 1]
        inv_sig_init = math.log(init_alpha / (1 - init_alpha)) if 0 < init_alpha < 1 else 0.0
        # We can make alpha per-channel for more flexibility!
        self.alpha_logits = nn.Parameter(torch.full((out_channels, 1, 1), inv_sig_init, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization as standard in PyTorch Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        x: [batch_size, in_channels, H, W]
        """
        # 1. MAC computation (Standard Convolution)
        y_mac = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        
        # 2. MAM computation (Max-Filtering)
        # To compute max(w * x) over the kernel patch, we can unfold the input
        # into patches, multiply by weights, and take the max.
        
        batch_size, in_channels, h_in, w_in = x.size()
        
        # Unfold input to patches: 
        # [batch_size, in_channels * kH * kW, L] where L is number of patches
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        L = x_unfolded.size(-1)
        
        # Reshape to [batch_size, 1, in_channels * kH * kW, L] for broadcasting
        x_unfolded = x_unfolded.view(batch_size, 1, -1, L)
        
        # Reshape weights to [out_channels, in_channels * kH * kW] then add dims: [1, out_channels, C*K*K, 1]
        w_flat = self.weight.view(self.out_channels, -1).unsqueeze(0).unsqueeze(-1)
        
        y_mam_list = []
        for o in range(self.out_channels):
            # w_o: [1, 1, C*K*K, 1]
            w_o = w_flat[:, o:o+1, :, :]
            # Element-wise product broadcast to [batch_size, 1, C*K*K, L]
            # Max over dim=2 computes MAM -> [batch_size, 1, L]
            out_o = torch.max(x_unfolded * w_o, dim=2)[0]
            y_mam_list.append(out_o)
            
        y_mam = torch.cat(y_mam_list, dim=1) # [batch_size, out_channels, L]
        
        # Fold it back to spatial dimensions
        h_out = y_mac.size(2)
        w_out = y_mac.size(3)
        y_mam = y_mam.view(batch_size, self.out_channels, h_out, w_out)
        
        if self.bias is not None:
            y_mam = y_mam + self.bias.view(1, -1, 1, 1)
            
        # 3. Combine MAC and MAM using adaptive parameter
        alpha = torch.sigmoid(self.alpha_logits) 
        
        y_out = alpha * y_mac + (1 - alpha) * y_mam
        
        return y_out

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias is not None
        )
