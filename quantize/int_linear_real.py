import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from quantize.triton_utils.kernels import dequant_dim0, dequant_dim1
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from tqdm import tqdm
import gc  
from quantize.utils import get_named_linears,set_op_by_name

logger = getLogger(__name__)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        **kwargs
    ):
        super().__init__()
        # if bits not in [2, 4, 8]:
        #     raise NotImplementedError("Only 2,4,8 bits are supported.")
        # if infeatures % 32 != 0 or outfeatures % 32 != 0:
        #     raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer(
            'qweight',
            torch.zeros((math.ceil(infeatures / (32 // self.bits)), outfeatures), dtype=torch.int32)
        )
        self.register_parameter(
            'scales',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(infeatures / self.group_size), math.ceil(outfeatures / (32 // self.bits))), dtype=torch.int32)
        )
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )   # not used, just for consistent with GPTQ models
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.use_fake = False

    def post_init(self):
        pass


    def use_fake_quantization(self, del_quant=False,transpose=False):
        # use fake quantization for faster training but consume more memory
        weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        dim0, dim1 = weight.shape
        zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
        weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1).contiguous()
        self.register_buffer(
            'weight',
            weight
        )
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.qzeros
            del self.g_idx
        
    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
    
        g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32)

        scale_zeros = zeros * scales
        self.scales = nn.Parameter(scales.half())
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx] + scale_zeros[g_idx[idx]]) / self.scales[g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((math.ceil(intweight.shape[0]/(32//self.bits)), intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 3, 4, 5, 6, 8]:
                for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,5,6,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros = zeros.numpy().astype(np.uint32)
        self.zeros_dim0, self.zeros_dim1 = zeros.shape
        qzeros = np.zeros((zeros.shape[0], math.ceil(zeros.shape[1] / (32 // self.bits))), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 3, 4, 5, 6, 8]:
                for j in range(i, min(i + (32 // self.bits), zeros.shape[1])):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,5,6,8 bits are supported.")
                
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
            dim0, dim1 = weight.shape
            # dim2 = (dim1*dim0)//self.group_size
            zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
            weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        # out = torch.matmul(x, weight)
        out = torch.matmul(x, weight.to(x.dtype))
        out = out + self.bias if self.bias is not None else out
        return out


def load_quantized_model(model_path, wbits, group_size):
    print(f"Loading quantized model from {model_path}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer


def load_mixed_precision_quantized_model(model_path):
    """
    Load a mixed-precision quantized model where different layers may have different bit-widths.
    This function reads the per-layer bit-widths and group sizes from the saved checkpoint.
    
    Args:
        model_path: Path to the saved mixed-precision quantized model
        
    Returns:
        model: Loaded model with per-layer quantization configurations
        tokenizer: Loaded tokenizer
    """
    import os
    print(f"Loading mixed-precision quantized model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    
    # Load state dict to inspect per-layer configurations
    state_dict_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(state_dict_path):
        # Try safetensors format
        state_dict_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(state_dict_path):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            # Try sharded checkpoints
            from transformers.modeling_utils import load_sharded_checkpoint
            state_dict, _ = load_sharded_checkpoint(model_path)
    else:
        state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # Extract per-layer bit-widths and group sizes from state dict
    layer_configs = {}
    for key in state_dict.keys():
        # Look for qweight keys to identify quantized layers
        # Format: model.layers.{layer_idx}.{submodule}.{linear_name}.qweight
        if 'qweight' in key:
            parts = key.split('.')
            if 'layers' in parts:
                layer_idx = int(parts[parts.index('layers') + 1])
                if layer_idx not in layer_configs:
                    layer_configs[layer_idx] = {}
                
                # Get the module path (e.g., "self_attn.q_proj")
                module_path_parts = []
                start_collecting = False
                for i, part in enumerate(parts):
                    if start_collecting and part not in ['qweight', 'scales', 'qzeros', 'g_idx', 'bias']:
                        module_path_parts.append(part)
                    if part == parts[parts.index('layers') + 1]:  # After layer index
                        start_collecting = True
                module_path = '.'.join(module_path_parts)
                
                # Infer bits from qweight shape
                # qweight shape: (math.ceil(infeatures / (32 // bits)), outfeatures)
                qweight_shape = state_dict[key].shape
                scales_key = key.replace('qweight', 'scales')
                if scales_key in state_dict:
                    scales_shape = state_dict[scales_key].shape
                    # scales shape: (math.ceil(infeatures / group_size), outfeatures)
                    infeatures_groups = scales_shape[0]
                    qweight_rows = qweight_shape[0]
                    
                    # Calculate bits: qweight_rows = math.ceil(infeatures / (32 // bits))
                    # Try common bit-widths: 2, 3, 4, 5, 6, 8
                    for candidate_bits in [2, 3, 4, 5, 6, 8]:
                        # Calculate what infeatures would be
                        expected_infeatures = qweight_rows * (32 // candidate_bits)
                        # Check if this matches with scales
                        for candidate_gs in [64, 128, 256]:
                            if infeatures_groups == math.ceil(expected_infeatures / candidate_gs):
                                layer_configs[layer_idx][module_path] = {
                                    'bits': candidate_bits,
                                    'group_size': candidate_gs,
                                    'infeatures': expected_infeatures,
                                    'outfeatures': qweight_shape[1]
                                }
                                break
                        if module_path in layer_configs[layer_idx]:
                            break
    
    print(f"Detected configurations for {len(layer_configs)} layers")
    if layer_configs:
        bits_summary = {}
        for layer_idx, modules in layer_configs.items():
            for module_path, cfg in modules.items():
                bits = cfg['bits']
                bits_summary[bits] = bits_summary.get(bits, 0) + 1
        print(f"Bit-width distribution: {bits_summary}")
    
    # Create model with empty weights
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16, trust_remote_code=True)
    
    # Replace Linear layers with QuantLinear using detected configurations
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Replacing with QuantLinear"):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        
        for name, module in named_linears.items():
            # Get configuration for this specific layer and module
            if i in layer_configs and name in layer_configs[i]:
                cfg = layer_configs[i][name]
                wbits = cfg['bits']
                group_size = cfg['group_size']
                print(f"  Layer {i}.{name}: {wbits}-bit, group_size={group_size}")
            else:
                # Fallback to default if not found (shouldn't happen for properly saved models)
                wbits = 4
                group_size = 128
                print(f"  Layer {i}.{name}: Using default {wbits}-bit, group_size={group_size} (config not found)")
            
            q_linear = QuantLinear(
                wbits, 
                group_size, 
                module.in_features, 
                module.out_features, 
                not module.bias is None
            )
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    
    # Load checkpoint with proper device map
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model, checkpoint=model_path, device_map=device_map, offload_state_dict=True)
    print("Mixed-precision quantized weights loaded successfully!")
    
    return model, tokenizer


__all__ = ["QuantLinear", "load_quantized_model", "load_mixed_precision_quantized_model"]
