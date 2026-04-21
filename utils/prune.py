import torch
import torch.nn.utils.prune as prune

def compute_gradient_saliency(model, criterion, data, target):
    """
    Computes importance scores for weights based on |weight * grad|.
    """
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    saliency = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.grad is not None:
                saliency[name] = torch.abs(module.weight.data * module.weight.grad.data)
    
    model.zero_grad()
    return saliency

def apply_oneshot_gradient_pruning(model, saliency_dict, prune_ratio):
    """
    Applies one-shot pruning keeping (1-prune_ratio) of the most salient weights globally.
    """
    # Collect all saliency scores
    all_scores = []
    for name, scores in saliency_dict.items():
        all_scores.append(scores.view(-1))
    
    if not all_scores:
        return
        
    all_scores = torch.cat(all_scores)
    
    # Determine the threshold for pruning
    num_params = all_scores.numel()
    num_prune = int(num_params * prune_ratio)
    
    if num_prune == 0:
        return
        
    threshold, _ = torch.topk(all_scores, num_prune, largest=False)
    threshold = threshold[-1] # The largest value among the smallest 'num_prune' values
    
    # Apply masking
    for name, module in model.named_modules():
        if name in saliency_dict:
            scores = saliency_dict[name]
            mask = (scores > threshold).float()
            # We use custom_from_mask to leverage PyTorch's pruning infrastructure
            # This handles replacing module.weight with module.weight_orig and a mask buffer
            prune.custom_from_mask(module, name='weight', mask=mask)

def get_sparsity(model):
    """
    Returns the percentage of zero weights in the model (for layers that have 'weight').
    """
    total_zeros = 0
    total_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            total_zeros += torch.sum(module.weight == 0).item()
            total_params += module.weight.nelement()
            
    if total_params == 0:
        return 0.0
    return float(total_zeros) / total_params

def remove_pruning_reparameterization(model):
    """
    Makes pruning permanent by removing the weight_orig and weight_mask buffers.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
