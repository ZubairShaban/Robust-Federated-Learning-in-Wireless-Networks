from operator import length_hint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
# import matplotlib.pyplot as plt
# import tensorflow as tf
from torch.utils.data.dataset import Dataset
import numpy as np
from datetime import datetime
import torchvision
# print(torchvision.__version__)
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import logging
import os
import random
import math
import syft as sy
from functions import mnistIID, mnistnon_IID, FedDataset, getImage
from utils4Simpulsive import averageModels

Ps = 2 # signal power

# # print(key)
alpha_list=[]
# key_array = np.array(key_n)

accu = []

# def Wrapper(batch_size, lr, no_of_epoch, no_of_clients, no_of_rounds, key, key_array, Ps):
def Wrapper():
    count = 0
    print("yes")

    args = {
        'batch_size': 64,
        'test_batch_size': 1000,
        'lr': 0.003,
        'log_interval': 10,
        'epochs': 3,
        'clients': 30,
        'seed':1,
        'rounds': 100,
        'C': 1,
        'lowest_snr': 20,
        # 'highest_snr': 20,
        'lowest_csi': 0,
        'highest_csi': 1,
        'drop_rate': 0,
        'images': 60000,
        'datatype': 'iid',
        'use_cuda': False,
        'save_model': True
    }

    use_cuda = args['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    hook = sy.TorchHook(torch)
    clients = []
    # snr_val = 10**(snr/10)
    # std = math.sqrt(Ps/snr_val)n                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         b

    for i in range(args['clients']):
        clients.append({'hook': sy.VirtualWorker(
            hook, id="client{}".format(i+1))})

    # print(clients)
    # os.chdir("/content/drive/MyDrive/FL_ZaaPoo/data/MNIST/raw")
    nUsers = 30
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # transform=transforms.ToTensor()
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    k = len(set(mnist_testset.targets.numpy()))
    # print(k)
    if(args['datatype'] == 'iid'):
        print("iid")
        # dictionary containing dictionary for 20 clients
        train_group = mnistIID(mnist_trainset, nUsers)
        test_group = mnistIID(mnist_testset, nUsers)
        print(len(train_group[1]))
        print(len(test_group[1]))
    elif(args['datatype'] == 'non_iid'):
        print("non_iid")
        train_group = mnistnon_IID(mnist_trainset, nUsers)
        test_group = mnistnon_IID(mnist_testset, nUsers)
        print(len(train_group[1]))
        # print(len(test_group[1]))

    for inx, client in enumerate(clients):
        trainset_id_list = list(train_group[inx])
        client['mnist_trainset'] = getImage(
            mnist_trainset, trainset_id_list, args['batch_size'])
        client['mnist_testset'] = getImage(
            mnist_testset, list(test_group[inx]), args['batch_size'])
        client['samples'] = len(trainset_id_list)/args['images']
        client['previousparam'] = 0
        client['globalparam'] = 0
        # client['curr'] = 0
        client['Evalue'] = 0
        client['previous2'] = 0

    #=================Global Model===================#
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    global_test_dataset = datasets.MNIST(
        './', train=False, download=True, transform=transform)
    global_test_loader = DataLoader(
        global_test_dataset, batch_size=args['batch_size'], shuffle=True)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = Func.relu(self.conv1(x))
            x = Func.max_pool2d(x, 2, 2)
            x = Func.relu(self.conv2(x))
            x = Func.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = Func.relu(self.fc1(x))
            x = self.fc2(x)
            return Func.log_softmax(x, dim=1)

    x_dict = {}
    y_dict = {}
    snr_value = args['lowest_snr']
    for c in range(args['clients']+1):
        dict_key = "client" + str(c)
        x_val = random.random()
        y_val = random.random()
        # snr_value = random.randint(args['lowest_snr'], args['highest_snr'])
    
        # snr_dict[dict_key] = snr_value
        x_dict[dict_key] = x_val
        y_dict[dict_key] = y_val


    def train(args, client, device, Ps,snr_value):

        cStatus = True
        client['model'].train()
        client['model'].send(client['hook'])
        
        print("client_ID", client['hook'].id)

        # snr = snr_value
        print("SNR==", snr_value)

        snr_val = 10**(snr_value/10)

        std = math.sqrt(Ps/snr_val)

        x = random.random()
        y = random.random()
        x = x_dict[client['hook'].id]
        y = y_dict[client['hook'].id]
        #x = random.random()
        #y = random.random()
        h = complex(x, y)
        print("Client:", client['hook'].id)
        # print("CSI", abs(h)/(std*std))

        
        # wireless channel needs to be considered
        # no noise in downlink
  
        # cStatus = True     # Client status
        for epoch in range(1, args['epochs']+1):
            for batch_idx, (data, target) in enumerate(client['mnist_trainset']):
                data = data.send(client['hook'])
                target = target.send(client['hook'])
                
                data, target = data.to(device), target.to(device)

                # print("data size:     ",data.size())
                
                client['optimizer'].zero_grad()
                output = client['model'](data)
                loss = Func.nll_loss(output, target)
                loss.backward()
                # print(loss.grad)
                client['optimizer'].step()

                # print("==========ye chalega kya========================")
                if batch_idx % args['log_interval'] == 0:
                    loss = loss.get()
                    # print('Model {} Train Epoch: {}'.format(client['hook'].id,epoch))
                    print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        client['hook'].id,
                        epoch, batch_idx *
                        args['batch_size'], len(
                            client['mnist_trainset']) * args['batch_size'],
                        100. * batch_idx / len(client['mnist_trainset']), loss.item()))

        client['model'].get()

        y_out = client['model'].conv1.weight
        client['model'].conv1.weight.data = y_out

        y_out = client['model'].conv2.weight
        # print("size of 2nd layer", y_out.size())

        yy = y_out - client['previousparam']

        print("=====================================================================")
        # print(yy)

        y1 = torch.flatten(yy)
        
        yTy = 0
        for i in range(list(y1.size())[0]):
            yTy = yTy + y1[i]*y1[i]

        # print('-----------')
        # print("xTTTTTTTTTTTTx: ", yTy)
        # print(yTy)

        client['Evalue'] = yTy
        # client['previousparam'] = yy
        # client['curr'] = yy
        # client['previousparam'] = y_out
        client['model'].conv2.weight.data = yy
        
        return cStatus

    def test(args, model, device, test_loader, count):
        # print("TEST SET PRDEICTION")
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # add losses together
                test_loss += Func.nll_loss(output,
                                           target, reduction='sum').item()

                # get the index of the max probability class
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss for model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        accu.append(100. * correct / len(test_loader.dataset))

    logging.info("Starting training !!")

    torch.manual_seed(args['seed'])
    global_model = CNN()

    for client in clients:
        torch.manual_seed(args['seed'])
        client['model'] = CNN().to(device)
        # print(client)
        client['optimizer'] = optim.SGD(
            client['model'].parameters(), lr=args['lr'])

    for fed_round in range(args['rounds']):

        print(fed_round)
        
        # number of selected clients
        client_good_channel = []
        Evalue_arr = []

        # at least 1 client is selected for training
        m = int(max(args['C'] * args['clients'], 1))

        # Selected devices
        np.random.seed(fed_round)
        # dont choose same client more than once
        selected_clients_inds = np.random.choice(
            range(len(clients)), m, replace=False)
        selected_clients = [clients[i] for i in selected_clients_inds]

        # Active devices
        np.random.seed(fed_round)
        active_clients_inds = np.random.choice(selected_clients_inds, int(
            (1-args['drop_rate']) * m), replace=False)  # drop clients
        active_clients = [clients[i] for i in active_clients_inds]
        print(len(active_clients_inds))

        snr = snr_value
        print("SNR==", snr)
        snr_val = 10**(snr/10)
        std = math.sqrt(Ps/snr_val)

        K_clients = len(active_clients_inds)
        
        if(fed_round == 0):
            t = torch.nn.init.normal_(client['model'].conv2.weight,0,std)
            # print(t.shape)
            count = 0
            for client in active_clients:
                client['previousparam'] = t
                count = count+1
                client['previous2'] = t
                # print(count)
        
        if(fed_round == 5): #randomise round -- adding impulsive noise in random round
            a0 = 0
            a1 = 1
            client['previousparam'] = client['previous2']

        for client in active_clients:
            print("train")

            good_channel = train(args, client, device, Ps,snr_value)
            
            if(good_channel == True):
                client_good_channel.append(client)
                Evalue_arr.append(client['Evalue'])
                
                # print("Output of model - -------------" ,client['model'])

        # print("Client max alpha banane wali value" , client['Evalue'])
        print('Evalue', Evalue_arr)

        E_max = max(Evalue_arr)
        alpha = Ps/E_max
        
        value = alpha.item()

        print(value) 
        alpha_list.append(value)
        print('alpha value', alpha)

        print("Clients with good channel are considered for averaging")

        for no in range(len(client_good_channel)):
            print(client_good_channel[no]['hook'].id)

            y_out = client['model'].conv2.weight
            y_out = y_out*(math.sqrt(alpha))
            
            client['model'].conv2.weight.data = y_out
        
        print()
        print("reached this step")

        global_model = averageModels(global_model, client_good_channel, snr_value, Ps,alpha,K_clients,fed_round)
        
        y_out = global_model.conv2.weight


        y_out = y_out/(math.sqrt(alpha)*K_clients)

        y_out_flat = torch.flatten(y_out)
        yTensor = 0
        for i in range(list(y_out_flat.size())[0]):
            yTensor = yTensor + y_out_flat[i]*y_out_flat[i]
        print('------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-----')
        print("xTTTTTTTTTTTTx: ", yTensor)


        current = y_out + client['previousparam']

        if(fed_round == 1):
            client['previous2'] = current

        client['previous2']  = client['previousparam']
        client['previousparam'] = current

        # print('global average model', globl.parameters())
        # Testing the average model
        test(args, global_model, device, global_test_loader, count)                             


        #print("Total Power =", power_1)
        print()

        for client in clients:
            client['model'].load_state_dict(global_model.state_dict())

    if (args['save_model']):
        torch.save(global_model.state_dict(), "FederatedLearning.pt")

    print("============ Accuracy ===========")
    # print(accu)
    return accu

accuracy1 = Wrapper()
print(accuracy1)
print("alpha_list")
print(alpha_list)