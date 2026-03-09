from operator import length_hint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from torch.utils.data import Dataset
import copy
from torch.utils.data.dataset import Dataset
import numpy as np
from datetime import datetime
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import logging
import os
import pickle as pkl
import random
import math
from scipy.stats import rayleigh

import argparse  # Import argparse
from hetero_epochs import GenerateLocalEpochs
from utils4main import averageModels
from models_utils import read_data, read_user_data, read_full_data
from data.Femnist.data_generator import generate_data as femnist_data_generator
from data.CIFAR10.data_generator import generate_data as cifar10_data_generator

# dataset = "Femnist"
dataset="CIFAR10"
def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning with Proximal Term')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train')
    parser.add_argument('--sys_hetero', type=int, default=0, help='percentage of clients with varying epochs')
    parser.add_argument('--no_of_clients', type=int, default=30, help='number of clients')
    parser.add_argument('--min_no_of_clients', type=int, default=20, help='minimum number of clients')
    parser.add_argument('--seed', type=int, default=4, help='random seed')
    parser.add_argument('--rounds', type=int, default=150, help='number of communication rounds')
    parser.add_argument('--C', type=float, default=1, help='fraction of clients to use for training')
    parser.add_argument('--snr_dB', type=int, default=0, help='signal-to-noise ratio in dB')
    parser.add_argument('--lowest_csi', type=float, default=0, help='lowest channel state information')
    parser.add_argument('--highest_csi', type=float, default=1, help='highest channel state information')
    parser.add_argument('--drop_rate', type=float, default=0, help='dropout rate of clients')
    parser.add_argument('--images', type=int, default=50000, help='number of images in dataset')
    parser.add_argument('--data_set', type=str, default='CIFAR10', help='dataset to use')
    parser.add_argument('--datatype', type=str, default='iid', help='data distribution type')
    parser.add_argument('--noise', type=bool, default=False, help='whether to add noise')
    # parser.add_argument('--intensity', type=float, default=1, help='noise intensity')
    parser.add_argument('--precoding', type=bool, default=False, help='whether to use precoding')
    parser.add_argument('--fading', type=bool, default=False, help='whether to use fading')
    parser.add_argument('--h_min', type=float, default=0.0, help='minimum channel gain')
    parser.add_argument('--prox', type=bool, default=False, help='whether to use proximal term')
    parser.add_argument('--prox_lambda', type=float, default=0.4, help='proximal term lambda')
    parser.add_argument('--noisyComm', type=bool, default=False, help='robust FL with noisy comm paper')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use CUDA')
    parser.add_argument('--save_model', type=bool, default=True, help='whether to save the model')
    parser.add_argument('--similarity', type=float, default=1, help='similarity factor for data distribution')
    parser.add_argument('--P', type=float, default=1, help='Power')

    return parser.parse_args()

test_loss_list=[]
alpha_list=[]
accu = []
global_training_loss_list=[]
def Wrapper(args):
        
    data = read_data(args.data_set,args.no_of_clients,args.similarity)
    total_users = len(data[0])
    print("totalusers",total_users)
    train_full, test_full = read_full_data(total_users,data,args.data_set)
    global_test_loader = DataLoader(test_full,args.batch_size)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    #device="cpu"

    clients = [{'id': f'client{i+1}'} for i in range(args.no_of_clients)]

    for inx, client in enumerate(clients):
        id, train, test = read_user_data(inx, data, dataset)
        client['train_dataset'] = DataLoader(train, args.batch_size)
        client['test_dataset'] = DataLoader(test, args.batch_size)
        client['samples'] = len(train)
        client['previousparam'] = 0
        client['globalparam'] = 0
        client['proximal_list']=[]
        client['curr'] = 0
        client['Evalue'] = 0
        client['csi']=0.0

    class Femnist_CNN(nn.Module):
        def __init__(self):
            super(Femnist_CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 47)

        def forward(self, x):
            x = x.view(-1, 1, 28, 28)
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    class CIFAR_CNN1(nn.Module):
            def __init__(self):
                super(CIFAR_CNN1, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 3 input channels (RGB images), 32 output channels, 5x5 kernel, stride 1
                self.conv2 = nn.Conv2d(32,64, 3, padding=1)  # 32 input channels, 64 output channels, 5x5 kernel, stride 1
                self.conv3 = nn.Conv2d(64,128,3, padding=1) # 64 input channels, 128 output channels, 5x5 kernel, stride 1
                self.fc1 = nn.Linear(128* 4 * 4, 256) # Adjust the input size based on the output of the last conv layer
                # self.fc1 = nn.Linear(1152, 256) # Adjust the input size based on the output of the last conv layer
                self.fc2 = nn.Linear(256,10)
            

            def forward(self, x):
                x = torch.reshape(x, (-1, 3, 32, 32)) #added
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv3(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 128 * 4 * 4)  # Adjust the input size based on the output of the last conv layer
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)
    
    def train(args, client,local_epochs, device):

        cStatus = True
        client['model'].train()
        gradient_norms = []


        for epoch in range(1, local_epochs+1):
            correct = 0
            total_loss=0.0
            total_batches=0
            for batch_idx, (data, target) in enumerate(client['train_dataset']):
             
                
                data, target = data.to(device), target.to(device)
                
                client['optimizer'].zero_grad()
                output = client['model'](data)
                loss=F.nll_loss(output,target)
                # loss.backward(retain_graph=True)
                # total_gradient_norm = 0.0
                # for param in client['model'].parameters():
                #    if param.grad is not None:
                #      total_gradient_norm += (param.grad.norm(2) ** 2).item()
                # gradient_norms.append(total_gradient_norm)
                
                if args.noisyComm==True:
                    base_loss = F.nll_loss(output, target)
                    sigma_sq=torch.sqrt(args.P / (10**(torch.tensor(args.snr_dB, device=device, dtype=torch.float32) / 10)))**2
                    # client['model'].zero_grad()
                    # base_loss.backward(retain_graph=True)
                    grad_norm_sq = sum((param.grad.norm()**2 for param in client['model'].parameters() if param.grad is not None))
                    loss = base_loss + sigma_sq * grad_norm_sq
                    # loss.backward()

                ####My new implementation of fedprox
                elif args.prox == True:
                    # proximal_term = torch.tensor(0.0, device=device, dtype=torch.float64)
                    # for name, w in client['model'].named_parameters():
                    #     w_t = global_model.state_dict()[name].to(w.device, dtype=torch.float64)
                    #     proximal_term += ((w - w_t).norm(2))
                    proximal_term = torch.tensor(0.0,device=device )
                    # print(f"Proximal Term requires_grad: {proximal_term.requires_grad}")

                    # iterate through the current and global model parameters
                    for w, w_t in zip(client['model'].parameters(), global_model.parameters()) :
                        w_t = w_t.detach()  # Detach global model parameters to avoid gradient tracking
                        # print(f"w.requires_grad: {w.requires_grad}, w_t.requires_grad: {w_t.requires_grad}") 
                        proximal_term += ((w-w_t).norm(2))**2
                    proximal_term = proximal_term.detach()
                    # print(f"Proximal Term requires_grad after loop: {proximal_term.requires_grad}")

                    # print("proximal term calculated",proximal_term)
                    # print(f"Proximal Term: {proximal_term.item()}")
                    # print(f"Proximal Loss Contribution: {(args.prox_lambda / 2) * proximal_term.item()}")
                    # print("proximal term requires grad",)
                    # base_loss=F.nll_loss(output,target)
                    # proximal_loss_contribution = (args.prox_lambda / 2) * proximal_term

                    loss=F.nll_loss(output,target)+ (args.prox_lambda/2)*proximal_term
                    # print(f"Base Loss: {base_loss.item():.6f}, Proximal Term: {proximal_term.item():.6f}, Proximal Loss Contribution: {proximal_loss_contribution.item():.6f},final loss: {loss.item():.6f}")

                loss.backward()
                total_gradient_norm = 0.0
                for param in client['model'].parameters():
                   if param.grad is not None:
                     total_gradient_norm += (param.grad.norm(2) ** 2).item()
                gradient_norms.append(total_gradient_norm)
                
                client['optimizer'].step()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum()
                total_loss+=loss.item()
                total_batches+=1

                if batch_idx % args.log_interval == 0:
                    loss = loss
                    print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        client['id'],
                        epoch, batch_idx *
                        args.batch_size, len(
                            client['train_dataset']) * args.batch_size,
                        100. * batch_idx / len(client['train_dataset']), loss.item()))
                
            print('Training Accuracy: {:.6f}'.format(100. * correct / len(client['train_dataset'].dataset)))
                  
            avg_loss=total_loss/total_batches
        # proximal_term_list.append(proximal_term) 
        if args.prox==True:
            client['proximal_list'].append(proximal_term.item())
        else:
            pass


        client_model=client['model'].state_dict()
        client['train_loss']=avg_loss
####Zubairs alpha implementation acoording to alternate way of cotaf
        avg_squared_gradient_norm = sum(gradient_norms) / len(gradient_norms)
        G_squared = avg_squared_gradient_norm
        denom=(args.epochs**2)*(args.lr**2)*G_squared
        client['Evalue']=denom
###Zubairs alpha implementation according to eq 10 of cotaf
        # expectation = {}
        # for k in client_model.keys():
        #    expectation[k] = client_model[k] - client['previousparam'][k]
        # squared_norm = sum(torch.norm(expectation[k], p=2) ** 2 for k in expectation.keys())
        # # print("sq norm",squared_norm)
        # exp_squared_norm=squared_norm*client['samples']
        # # print("expected sq norm",exp_squared_norm)
        # client['Evalue']=exp_squared_norm
        
        return cStatus,client_model

    def test(args, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # add losses together
                test_loss += F.nll_loss(output,
                                           target, reduction='sum').item()

                # get the index of the max probability class
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest loss for global model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    

        accu.append(100. * correct / len(test_loader.dataset))
        test_loss_list.append(test_loss)

    if args.data_set == "Femnist" :
        for client in clients:
           torch.manual_seed(args.seed)
           client['model'] = Femnist_CNN().to(device)
           client['optimizer'] = optim.Adam (
            client['model'].parameters(), lr=args.lr)
        
        global_model = Femnist_CNN().to(device)
    if dataset == "CIFAR10" :
        for client in clients:
          torch.manual_seed(args.seed)
          client['model'] = CIFAR_CNN1().to(device)
          client['optimizer'] = optim.Adam (
            client['model'].parameters(), lr=args.lr)
        
        global_model = CIFAR_CNN1().to(device)



    h_min=rayleigh.ppf(1-(args.min_no_of_clients/args.no_of_clients))##new calculated h_min
    args.h_min = h_min
    for fed_round in range(args.rounds):

        print("Communication round ===================================================================",fed_round)
        
        # number of selected clients
        client_good_channel = []
        Evalue_arr = []
        clients_models=[]
        training_loss_arr=[]

       
        np.random.seed(fed_round)
        m = int(max(args.C * args.no_of_clients, 1))
        
        selected_clients_inds = np.random.choice(
                range(len(clients)), m, replace=False)
        participating_clients = [clients[i] for i in selected_clients_inds]
        
        if args.fading==True:
            active_clients = []
            # selected_clients_inds=[]
            for client in participating_clients:
                client['csi']=rayleigh.rvs()
            for i,client in enumerate(participating_clients):
                if client['csi']>=h_min:
                    # selected_clients_inds.append(i)
                    active_clients.append(client)
            print("Number of active clients",len(active_clients))
            K_clients = len(active_clients)
    
        else:
            print("Number of active clients",len(participating_clients))
            K_clients = len(participating_clients)
            active_clients=participating_clients

        # K_clients = len(active_clients_inds)
        local_epochs_list=GenerateLocalEpochs(args.sys_hetero,K_clients,args.epochs)
        print("all epoch list=",local_epochs_list)

        
        if (fed_round==0):

            # initial_model=initialize_weights(CNN())
            for client in clients:
              client['model'].load_state_dict(global_model.state_dict())
              client['previousparam'] = copy.deepcopy(client['model'].state_dict())
               
        idx=0        
        for client in active_clients:
            local_epochs=local_epochs_list[idx]
            print("Clients local epochs= ",local_epochs)
            idx+=1
            # global_model.send(client['hook'])   
            # if algorithm=="cotaf" and local_epochs<args.epochs:
            #     print("dropping straggler")
            #     continue
            good_channel = train(args, client,local_epochs, device)
            
            if(good_channel[0]==True):
                client_good_channel.append(client)#contains All info of each client
                clients_models.append(good_channel[1])#contains only state_dict() of each client
                Evalue_arr.append(client['Evalue'])
                training_loss_arr.append(client['train_loss'])
        # print("Evalues",Evalue_arr)
        E_max=max(Evalue_arr)
        alpha=args.P/E_max
        # alpha_value=alpha.item()
        alpha_value=alpha

        alpha_list.append(alpha_value)
        global_training_loss=sum(training_loss_arr)/len(training_loss_arr)
        global_training_loss_list.append(global_training_loss)

        
        # averaged_clients = []
        # for no in range(len(client_good_channel)):
        #     averaged_clients.append(client_good_channel[no]['hook'].id)

        #     y_out = client['model'].conv2.weight
        #     y_out = y_out*(math.sqrt(alpha))
            
        #     client['model'].conv2.weight.data = y_out
        # print("Clients with good channel are considered for averaging:", averaged_clients)
        global_model = averageModels(global_model, client_good_channel, args.P, alpha,K_clients,fed_round,args,h_min,device)
       
        test(args, global_model, device, global_test_loader)                             

        for client in clients:
            client['previousparam'] = copy.deepcopy(client['model'].state_dict())
            client['model'].load_state_dict(global_model.state_dict())



    print("============ Accuracy ===========")
    prox_list=[]
    for client in active_clients:
            prox_list.append(client['proximal_list'])
            # print("proximal term for client",client['proximal_list'])

    return accu,prox_list,args

if __name__ == "__main__":    
    args = parse_arguments()
    if args.noise==True and args.precoding==True and args.prox==True and args.noisyComm==False:     
            algorithm="norota"
    if args.noise==True and args.precoding==True and args.prox==False and args.noisyComm==False:   
        algorithm="cotaf"
    if args.noise==False and args.precoding==False and args.prox==True and args.noisyComm==False:  
        algorithm="fedprox"
    if args.noise==True and args.precoding==False and args.prox==True and args.noisyComm==False:
        algorithm="noisyprox"
    if args.noise==False and args.precoding==False and args.prox==False and args.noisyComm==False:
        algorithm="fedavg"
    if args.noise==True and args.precoding==False and args.prox==False and args.noisyComm==False:
        algorithm="noisyfedavg"  
    if args.noise==True and args.precoding==False and args.prox==False and args.noisyComm==True:
        algorithm="noisyComm"
    if args.fading==True:
        algorithm = f"{algorithm}F"
    print("Algorithm:",algorithm)
    accuracy1, prox_list, args = Wrapper(args)
    # print("global training loss=",global_training_loss_list)
    def format_snr(snr):
        return f"minus{abs(snr)}" if snr < 0 else str(snr)
    def format_similarity(similarity):
            return f"{str(similarity).replace('.', 'Pt')}"
    def format_lambda(prox_lambda):
            return f"{str(prox_lambda).replace('.', 'Pt')}"
    # def format_intensity(intensity):
    #         return f"{str(intensity).replace('.', 'Pt')}"
    def format_algo(args):
        if args.noise==True and args.precoding==True and args.prox==True and args.noisyComm==False:     
            algorithm="norota"
        if args.noise==True and args.precoding==True and args.prox==False and args.noisyComm==False:   
            algorithm="cotaf"
        if args.noise==False and args.precoding==False and args.prox==True and args.noisyComm==False:  
            algorithm="fedprox"
        if args.noise==True and args.precoding==False and args.prox==True and args.noisyComm==False:
            algorithm="noisyprox"
        if args.noise==False and args.precoding==False and args.prox==False and args.noisyComm==False:
            algorithm="fedavg"
        if args.noise==True and args.precoding==False and args.prox==False and args.noisyComm==False:
            algorithm="noisyfedavg"  
        if args.noise==True and args.precoding==False and args.prox==False and args.noisyComm==True:
            algorithm="noisyComm"
        return f"{algorithm}F" if args.fading==True else algorithm   
    
    algo_str=format_algo(args)
    snr_str = format_snr(args.snr_dB)
    similarity_str = format_similarity(args.similarity)
    lambda_str=format_lambda(args.prox_lambda)
    C_str=format_lambda(args.C)
    # intensity_str=format_intensity(args.intensity)
    base_filename = f"{algo_str}_{args.data_set}_clients{args.no_of_clients}_SNR{snr_str}_sim{similarity_str}_lambda{lambda_str}_sh{args.sys_hetero}_minclients{args.min_no_of_clients}_C{C_str}"
    i = 1
    filename = f"{base_filename}.csv"
    while os.path.exists(os.path.join("pseudo_results", filename)) and i < 5:
        i += 1
        filename = f"{base_filename}_{i}.csv"
        
    pseudo_results_dir="pseudo_results"
    os.makedirs(pseudo_results_dir, exist_ok=True)
    file_path = os.path.join(pseudo_results_dir, filename)


    # Save the sys_hetero_results to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the headers
        writer.writerow([f"Accuracy_{algo_str}_{args.data_set}_clients{args.no_of_clients}_SNR{snr_str}_sim{similarity_str}_lambda{lambda_str}_sh{args.sys_hetero}=" + str(accuracy1)])
        writer.writerow([f"Precoding_{algo_str}_{args.data_set}_clients{args.no_of_clients}_snr{snr_str}_sim{similarity_str}_lambda{lambda_str}_sh{args.sys_hetero}=" + str(alpha_list)])
        writer.writerow([f"Test_loss_{algo_str}_{args.data_set}_clients{args.no_of_clients}_snr{snr_str}_sim{similarity_str}_lambda{lambda_str}_sh{args.sys_hetero}=" + str(test_loss_list)])   
        writer.writerow([f"Training_loss_{algo_str}_{args.data_set}_clients{args.no_of_clients}_snr{snr_str}_sim{similarity_str}_lambda{lambda_str}_sh{args.sys_hetero}=" + str(global_training_loss_list)])   

        # writer.writerow(['Prox List=' + str(prox_list)]) 
        writer.writerow([f"Client_ProxList_{algo_str}_{args.data_set}_clients{args.no_of_clients}_snr{snr_str}_sim{similarity_str}_lambda{lambda_str}_sh{args.sys_hetero}=" + str(prox_list[1])])
        # all_clients_specific_round_proxterms = [sublist[10] for sublist in prox_list]
        # writer.writerow(['All clients specific round Prox List=' + str(all_clients_specific_round_proxterms)])

        # Write the arguments
        for key, value in vars(args).items():
            writer.writerow([key, value])

    print(f"Results saved successfully to {file_path}")