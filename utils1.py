import torch
import math


def averageModels(global_model, clients, snr_value, Ps,alpha,K_clients,fed_round):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    client_models = [clients[i]['model'] for i in range(len(clients))]
    
    samples = [clients[i]['samples'] for i in range(len(clients))]

    global_dict = global_model.state_dict()

    snr = snr_value
    # print("SNR==", snr)
    snr_val = 10**(snr/10)
    std = math.sqrt(Ps/snr_val)

    for k in global_dict.keys():  
        client_weights = [client_models[i].state_dict()[k].float() * samples[i] for i in range(len(client_models))]
        weighted_sum = torch.stack(client_weights, dim=0).sum(dim=0)
        global_dict[k] = weighted_sum

        # Add Gaussian noise to the global model's parameters
        # noise = torch.randn(global_dict[k].shape) * (std/(K_clients))
        noise = 0
        
        # if(fed_round == 25): #randomise round -- adding impulsive noise in random round
        #     a0 = 0
        #     a1 = 1
            
        # else:
        #     a0 = 1
        #     a1 = 0

        # print("Guassian value : ", a0)
        std1 = math.sqrt(Ps/(snr_val)) 
        # std2 = 50*std1

        # #std1 = math.sqrt(0.02/(a0+50*a1))
        # # print(Ps/(snr_val*(a0+50*a1)))

        # print("std1",std1)
        
        n1 = torch.randn(global_dict[k].shape)
        # n2 = torch.randn(global_dict[k].shape)
        # noise = a0*n1*std1 + a1*n2*std2
        noise=n1*std

        # print(noise.size())
        global_dict[k] += noise/(K_clients)
        
        # h = 1
        # y_out = global_model.conv2.weight

        # noise = torch.randn(y_out.size())


        # std1 = math.sqrt(Ps/(snr_val*(a0+50*a1))) 
        

        # y_out = y_out/(math.sqrt(alpha)*K_clients)
        
        # impulsive noise is added here
        # y_out = h*y_out + noise

    global_model.load_state_dict(global_dict)

    return global_model
