import torch
import math
import numpy as np
from typing import Any, Dict, List
import copy
from scipy.stats import rayleigh


def averageModels(global_model, client_good_channel, P,alpha, K_clients, fed_round,args,h_min,device):
     
# Assuming CNN() is already defined and client_good_channel is available
    client_models = [client_good_channel[i]['model'] for i in range(len(client_good_channel))]
    samples = [client_good_channel[i]['samples'] for i in range(len(client_good_channel))]

    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        # This is a simple average where samples[i] is not used; weights are divided by the number of clients, i.e., mean(dim=0)
        client_weights = [client_models[i].state_dict()[k].float().to(device) for i in range(len(client_models))]
        weighted_sum = torch.stack(client_weights, dim=0).mean(dim=0)
        global_dict[k] = weighted_sum

        std_dev = torch.sqrt(P / (10**(torch.tensor(args.snr_dB, device=device, dtype=torch.float32) / 10)))
        noise = torch.randn_like(global_dict[k], device=device) * std_dev

        # Divide noise tensor on GPU
        if args.noise==True and args.precoding==True:

            noise /= (K_clients * torch.sqrt(torch.tensor(alpha, device=device, dtype=torch.float32)))
            if args.fading==True:
                global_dict[k] = global_dict[k] + noise/h_min
            else:
                global_dict[k] = global_dict[k] + noise

        elif args.noise==True and args.precoding==False:
            noise/=K_clients
            # noise=noise
            global_dict[k] = global_dict[k]+noise
        else:
            global_dict[k] = global_dict[k]

    # print(global_dict)

    global_model.load_state_dict(global_dict)
    return global_model



