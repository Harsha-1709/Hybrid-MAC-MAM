import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridLinear(nn.Module):
    """
    Adaptive Hybrid MAC-MAM Layer.
    Combines standard Multiply-and-Accumulate (MAC) with Multiply-And-Max (MAM).
    The combination is controlled by a learnable parameter alpha.
    """
    def __init__(self, in_features, out_features, bias=True, init_alpha=0.5):
        super(HybridLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Learnable adaptive parameter alpha for balancing MAC and MAM
        # Sigmoid will be applied to keep it in [0, 1]
        # inv_sigmoid(init_alpha) roughly gives the starting value
        inv_sig_init = math.log(init_alpha / (1 - init_alpha)) if 0 < init_alpha < 1 else 0.0
        self.alpha_logits = nn.Parameter(torch.tensor(inv_sig_init, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization as standard in PyTorch Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        x: [batch_size, in_features]
        """
        batch_size = x.size(0)
        
        # 1. MAC computation
        # y_mac: [batch_size, out_features]
        y_mac = F.linear(x, self.weight, self.bias)
        
        # 2. MAM computation
        # Compute max(w_o * x) for each output neuron without huge intermediate tensor
        y_mam_list = []
        for o in range(self.out_features):
            w_o = self.weight[o:o+1, :] # [1, in_features]
            out_o = torch.max(x * w_o, dim=1, keepdim=True)[0] # [batch_size, 1]
            y_mam_list.append(out_o)
            
        y_mam = torch.cat(y_mam_list, dim=1) # [batch_size, out_features]

        
        if self.bias is not None:
            y_mam = y_mam + self.bias
            
        # 3. Combine MAC and MAM using adaptive parameter
        alpha = torch.sigmoid(self.alpha_logits)
        
        y_out = alpha * y_mac + (1 - alpha) * y_mam
        
        return y_out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
