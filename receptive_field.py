#script modified from Barnes EA et. al (2022) doi: 10.1175/AIES-D-22-0001.1 to include channel-specific prototypes

import numpy as np
import torch
import copy
import torch.nn as nn

def compute_rf(base, proto, input_size, conv_output_size):

    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device('cuda')
        print("GPU is set as the default device.")
    else:
        print("CUDA is not available. CPU will be used as the default device.")
        device = 'cpu'
    sample = np.zeros(shape=(1, *input_size[1:]), dtype=np.float32)
    
    cloned_base = copy.deepcopy(base)
    cloned_proto = copy.deepcopy(proto)

    cloned_base.to(device)
    cloned_proto.to(device)

    for p in cloned_base.base_layers:
        if isinstance(p, nn.Conv2d):
            p.bias = nn.Parameter(torch.zeros((p.bias.shape)))
            p.weight = nn.Parameter(torch.ones((p.weight.shape)))

    for p in cloned_proto.proto_layers:
        if isinstance(p, nn.Conv2d):
            p.bias = nn.Parameter(torch.zeros((p.bias.shape)))
            p.weight = nn.Parameter(torch.ones((p.weight.shape)))

    cloned_base.base_layers = nn.ModuleList(map(lambda x: nn.Identity() if isinstance(x, nn.LeakyReLU) else x, cloned_base.base_layers))
    cloned_proto.proto_layers = nn.ModuleList(map(lambda x: nn.Identity() if isinstance(x, nn.LeakyReLU) else x, cloned_proto.proto_layers))

    imin = np.full(shape=conv_output_size[2], fill_value=np.iinfo(np.int32).max, dtype=np.int32)
    imax = np.full(shape=conv_output_size[2], fill_value=-1,                dtype=np.int32)
    jmin = np.full(shape=conv_output_size[3], fill_value=np.iinfo(np.int32).max, dtype=np.int32)
    jmax = np.full(shape=conv_output_size[3], fill_value=-1,                dtype=np.int32)

    for i in range(input_size[2]):
        sample[0, 0, i, 0] = 1
        output = cloned_base(torch.from_numpy(sample).to(device))
        output, _ = cloned_proto.push_forward(output)
        sample[0, 0, i, 0] = 0  
        result = torch.amax(output, dim=(1)) 
        for m in range(output.shape[2]):
            if result[0, m, 0] != 0:
                imin[m] = min(imin[m], i)
                imax[m] = max(imax[m], i)


    for j in range(input_size[3]):
        sample[0, 0, 0, j] = 1
        output = cloned_base(torch.from_numpy(sample).to(device))
        output, _ = cloned_proto.push_forward(output)
        sample[0, 0, 0, j] = 0  

        result = torch.amax(output, dim=(1)) 
        for n in range(output.shape[3]):
            if result[0, 0, n] != 0:
                jmin[n] = min(jmin[n], j)
                jmax[n] = max(jmax[n], j)

    return imin, imax, jmin, jmax

def output_to_input(m,n,imin, imax, jmin, jmax, input_size):
    input_size = input_size[2:]
    input_mask = np.zeros(input_size)
    ones_shape = (imax[m]+1-imin[m],jmax[n]+1-jmin[n])
    input_mask[imin[m]:imax[m]+1, jmin[n]:jmax[n]+1] = np.ones(shape=ones_shape)
    return input_mask, imin[m], imax[m]+1, jmin[n], jmax[n]+1
