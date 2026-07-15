import torch
import torch.nn as nn
import math

# LoRA Implementation
class LoRALayer(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_linear, name)

    def forward(self, x, **kwargs):
        original_out = self.original_linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return original_out + (lora_out * self.scaling)


# Adapter Implementation
class AdapterLayer(nn.Module):
    def __init__(self, original_linear, reduction_factor=4):
        super().__init__()
        self.original_linear = original_linear
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features 
        
        bottleneck_dim = max(1, in_dim // reduction_factor)
        
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.adapter_down = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.adapter_up = nn.Linear(bottleneck_dim, out_dim)
        
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_linear, name)

    def forward(self, x, **kwargs):
        original_out = self.original_linear(x)
        adapter_out = self.adapter_down(x)
        adapter_out = self.act(adapter_out)
        adapter_out = self.adapter_up(adapter_out)
        return original_out + adapter_out


# Injection Helpers
def should_skip_layer(name):
    keywords = ['classifier', 'pooler', 'score', 'head', 'fc']
    return any(k in name for k in keywords)


def inject_lora(model, rank=4, alpha=16):
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_skip_layer(name):
                continue
            if isinstance(module, (LoRALayer, AdapterLayer)):
                continue

            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            modules_to_replace.append((parent_name, child_name, module))

    if not modules_to_replace:
        print("[LoRA] Warning: No suitable Linear layers found.")
        return model

    for parent_name, child_name, original_module in modules_to_replace:
        lora_layer = LoRALayer(original_module, rank=rank, alpha=alpha)
        if parent_name:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, lora_layer)
        else:
            setattr(model, child_name, lora_layer)
            
    print(f"[LoRA] Injected {len(modules_to_replace)} LoRA layers.")
    return model


def inject_adapter(model, reduction_factor=4):
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_skip_layer(name):
                continue
            if isinstance(module, (LoRALayer, AdapterLayer)):
                continue

            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            modules_to_replace.append((parent_name, child_name, module))

    if not modules_to_replace:
        print("[Adapter] Warning: No suitable Linear layers found.")
        return model

    for parent_name, child_name, original_module in modules_to_replace:
        adapter_layer = AdapterLayer(original_module, reduction_factor=reduction_factor)
        if parent_name:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, adapter_layer)
        else:
            setattr(model, child_name, adapter_layer)
            
    print(f"[Adapter] Injected {len(modules_to_replace)} Adapter layers.")
    return model


# Main Tuning Strategy Application
def apply_tuning_strategy(model, strategy, config=None):
    print(f"[Tuner] Applying strategy: {strategy}")
    
    for param in model.parameters():
        param.requires_grad = False
        
    head_found = False
    for name, param in model.named_parameters():
        if should_skip_layer(name):
            param.requires_grad = True
            head_found = True
            
    if not head_found:
        print("[Tuner] Warning: No classifier head detected to unfreeze!")
    
    if strategy == 'full_ft':
        for param in model.parameters():
            param.requires_grad = True
    elif strategy == 'head_only':
        pass
    elif strategy == 'lora':
        model = inject_lora(model, rank=4, alpha=16)
    elif strategy == 'adapter':
        model = inject_adapter(model, reduction_factor=4)
    else:
        raise ValueError(f"Strategy {strategy} not implemented.")
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable_params / all_params if all_params > 0 else 0
    
    trainable_str = "{:,}".format(trainable_params)
    all_str = "{:,}".format(all_params)
    print(f"[Tuner] Trainable Params: {trainable_str} / {all_str} ({ratio:.2f}%)")
    
    return model