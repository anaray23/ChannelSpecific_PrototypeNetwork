
from preprocess_data import Custom_Dataset, ChannelMNISTDataset
from torch.utils.data import DataLoader
import torch
from push_prototypes import push
import h5py
from receptive_field import compute_rf, output_to_input
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.optim.lr_scheduler as lr_scheduler
from torchgeo_euro import EuroSAT, ScaleAndNormalize
from torchvision.transforms import v2 as transforms
import torch.nn as nn 
from receptive_field import compute_rf, output_to_input
import pickle


if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device('cuda')
    print("GPU is set as the default device.")
else:
    print("CUDA is not available. CPU will be used as the default device.")
    device = 'cpu'


#set nc=True to run Prototype Network (No Channel-Specific Prototypes)
nc = False 
if nc == True:
    from model_channel_rs_nc import BaseConvModel, ProtoModel, FinalLayer, WholeModel, exp_var_dict
else:
    from model_channel_rs import BaseConvModel, ProtoModel, FinalLayer, WholeModel, exp_var_dict

#Model save info
model_save = False
#if model_save == True:
    #prototype_info = ''
    #model_name = ''
cross_loss = nn.CrossEntropyLoss()


#Training Parameters
base_epochs = 25
train_epochs = 50
exp_dataset = exp_var_dict['exp_dataset']

# Determines at which epoch to run Stage 2
def push_epoch(epoch, flag, delay_push_epoch = 10):
    if flag: #Project Stage 2 every training cycle
        return (epoch % 2 == 0)
    else: #Project Stage 2 as per delay_push_epoch parameter
        return (epoch % delay_push_epoch == 0)


#Load in dataset
if exp_dataset=='mjo':
    lead_time = 0
    #file_name = "data/processed_mjo_data/processed_mjo_noise.hdf5"
    file_name = "data/processed_mjo_data/processed_mjo.hdf5"

    train_data = Custom_Dataset(hdf5_data_file=file_name, im_name='train_images',label_name='train_labels', exp_dataset=exp_dataset)
    val_data = Custom_Dataset(hdf5_data_file=file_name, im_name='val_images',label_name='val_labels', exp_dataset=exp_dataset)
    test_data = Custom_Dataset(hdf5_data_file=file_name, im_name='test_images',label_name='test_labels', exp_dataset=exp_dataset)

    with h5py.File(file_name,'r') as f:
        all_images = f['train_images'][:]
        all_labels = f['train_labels'][:]
        all_times = f['train_time'][:]
        test_images = f['test_images'][:]
        test_labels = f['test_labels'][:]

    train_dataloader = DataLoader(train_data, batch_size = exp_var_dict['batch_size'])
    val_dataloader = DataLoader(val_data)
    test_dataloader = DataLoader(test_data, shuffle=False)
    push_dataloader = DataLoader(train_data, batch_size = len(all_images), shuffle=False)

elif exp_dataset == 'mnist':
    train_data = ChannelMNISTDataset(hdf5_data_file='data/syntheticMNIST/synthetic_data_channelmnist.hdf5', im_name='train_images',label_name='train_labels')
    val_data = ChannelMNISTDataset(hdf5_data_file='data/syntheticMNIST/synthetic_data_channelmnist.hdf5', im_name='val_images',label_name='val_labels')
    test_data = ChannelMNISTDataset(hdf5_data_file='data/syntheticMNIST/synthetic_data_channelmnist.hdf5', im_name='test_images',label_name='test_labels')

    with h5py.File("data/syntheticMNIST/synthetic_data_channelmnist.hdf5",'r') as f:
        all_images = f['train_images'][:]
        all_labels = f['train_labels'][:]
        test_images = f['test_images'][:]
        
    train_dataloader = DataLoader(train_data, batch_size = exp_var_dict['batch_size'])
    val_dataloader = DataLoader(val_data)
    test_dataloader = DataLoader(test_data)
    push_dataloader = DataLoader(train_data, batch_size = len(all_images), shuffle=False)


elif exp_dataset == 'euro_rs':
    root='data/EuroSAT/eurosat_data_train'
    min=[816.,   0.,   0.,   0., 174., 153., 128.,   0.,  40.,   1.,   5.,   1., 91.]
    max=[17720., 28000., 28000., 28000., 23381., 27791., 28001., 28002., 15384.,183., 24704., 22210., 28000.]

    mean = [1354.4054, 1118.2439, 1042.9299,  947.6262, 1199.4728, 1999.7909, 2369.2229, 2296.8262,  732.0834,   12.1133, 1819.0103, 1118.9240, 2594.1406]
    std = [ 245.7176,  333.0078,  395.0925,  593.7505,  566.4170,  861.1840, 1086.6315, 1117.9817,  404.9198,    4.7758, 1002.5877,  761.3033, 1231.5858]
    
    scale_and_normalize_transform = ScaleAndNormalize(mean, std)

    train_dataset = EuroSAT(root, split='train', download=False, transforms=scale_and_normalize_transform)
    val_dataset = EuroSAT(root, split='val', download=False, transforms=scale_and_normalize_transform)
    test_dataset = EuroSAT(root, split='test', download=False, transforms=scale_and_normalize_transform)
    generator = torch.Generator(device)

    train_dataloader = DataLoader(train_dataset, batch_size = exp_var_dict['batch_size'], generator=generator, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = exp_var_dict['batch_size'], generator=generator)
    test_dataloader = DataLoader(test_dataset, generator=generator)
    push_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle=False, generator=generator)

############## Train Standard Neural Network without Prototypes ##############################
base = BaseConvModel(exp_dataset, CNN_only = True)
base_optimizer = torch.optim.Adam([{'params':base.base_layers.parameters()}], lr=0.001)

optimizer = base_optimizer
for epoch in range(base_epochs):
    epoch_cross_entropy = 0
    tn_examples = 0
    tn_correct = 0
    for inputs, labels in train_dataloader:
        inputs = torch.nan_to_num(inputs)
        base.train()
        with torch.enable_grad():
            output = base(inputs)
            cross_entropy = cross_loss(output, labels.to(torch.long))
            output = torch.nn.functional.softmax(output)

            predicted = torch.max(output.data, 1)
            tn_examples += labels.size(0)
            tn_correct += torch.eq(predicted[1], labels).sum().item()

        loss = cross_entropy 
        epoch_cross_entropy += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_epoch_cross_entropy = 0
    n_examples = 0
    n_correct = 0
    for inputs, labels in val_dataloader:
        inputs = torch.nan_to_num(inputs)
        base.eval()
        with torch.no_grad():
            output = base(inputs)
            cross_entropy = cross_loss(output, labels.to(torch.long))
            output = torch.nn.functional.softmax(output)

            predicted = torch.max(output.data, 1)
            n_examples += labels.size(0)
            n_correct += torch.eq(predicted[1], labels).sum().item()
            
            val_epoch_cross_entropy+= cross_entropy.item()
    print('Epoch '+str(epoch)+' Train Accuracy: '+str(tn_correct/tn_examples))
    print('Epoch '+str(epoch)+' Val Accuracy: '+str(n_correct/n_examples))

true_label = []
pred_label = []
n_examples = 0
n_correct = 0
n_correct_1 = 0
for inputs, labels in test_dataloader:
    inputs = torch.nan_to_num(inputs)
    base.eval()
    with torch.no_grad():
        output = base(inputs)
        cross_entropy = cross_loss(output, labels.to(torch.long))
        output = torch.nn.functional.softmax(output)

        predicted = torch.max(output.data, 1)
        n_examples += labels.size(0)
        n_correct += torch.eq(predicted[1], labels).sum().item()
        n_correct_1 += (abs(predicted[1]-labels) < 2).sum().item()
        true_label.append(int(labels.cpu().item()))
        pred_label.append(predicted[1].cpu().item())
print('Test Accuracy: '+str(n_correct/n_examples))
print('+1 Test Accuracy: '+str(n_correct_1/n_examples))

############## Train Channel-Specific Prototype Network ##############################
#script modified from Barnes EA et. al (2022) doi: 10.1175/AIES-D-22-0001.1 and Chen C et al (2019) doi: 10.48550/arXiv.1806.10574 to include channel-specific prototypes

#whole = WholeModel(exp_dataset, exp_var_dict['num_channels'], backbone_weights=base.state_dict())
whole = WholeModel(exp_dataset, exp_var_dict['num_channels'])

#Create stage 1 and stage 3 optimizers
optimizer_specs = []
for base_chan in whole.base_list:
    optimizer_specs.append({'params':base_chan.base_layers.parameters()})
if exp_var_dict['location_scale'] == True:
    for proto_chan in whole.proto_list:
        optimizer_specs.append({'params':proto_chan.proto_layers.parameters()})
        optimizer_specs.append({'params':proto_chan.prototypes})
        optimizer_specs.append({'params':proto_chan.location_scale})
else:
    for proto_chan in whole.proto_list:
        optimizer_specs.append({'params':proto_chan.proto_layers.parameters()})
        optimizer_specs.append({'params':proto_chan.prototypes})
optimizer_specs.append({'params':whole.final_list[0].final_layers.parameters()})

stage1_optimizer = torch.optim.Adam(optimizer_specs[:-1], lr=exp_var_dict['stage1_lr'], weight_decay=1e-3)
stage3_optimizer = torch.optim.Adam([optimizer_specs[-1]], lr = exp_var_dict['stage3_lr'], weight_decay=1e-3)


num_train_epochs=train_epochs
validation_acc_arr = []
stage = 1


if exp_dataset == 'euro_rs': #reduce learning rate every 20 epochs for EuroSAT
    lr_step = 20
else:
    lr_step = 100
scheduler = lr_scheduler.StepLR(stage1_optimizer, step_size=lr_step, gamma=0.5)
scheduler3 = lr_scheduler.StepLR(stage3_optimizer, step_size=lr_step, gamma=0.5)

#Train Model and Validation
for epoch in range(num_train_epochs):
    print('Epoch: '+str(epoch))
    print('stage: '+str(stage))

    epoch_cross_entropy = 0
    epoch_cluster_cost = 0
    epoch_separation_cost = 0 
    epoch_div_cost = 0
    epoch_avg_separation_cost = 0
    train_labels_arr = []

    n_examples = 0
    n_correct = 0
    epoch_loss = 0

    #train
    for inputs, labels in train_dataloader:
        inputs = torch.nan_to_num(inputs)
        train_labels_arr.append(labels.cpu())

        whole.base_list[0].train()
        for chan in range(exp_var_dict['num_channels']):
            if exp_var_dict['sep_enc'] == True:
                whole.base_list[chan].train()
            whole.proto_list[chan].train()
        whole.final_list[0].train()
        with torch.enable_grad():

            #Stage 1
            if stage == 1:
                optimizer = stage1_optimizer
                for p in whole.base_list[0].parameters():
                    p.requires_grad = True
                for chan in range(exp_var_dict['num_channels']):
                    if exp_var_dict['sep_enc'] == True:
                        for p in whole.base_list[chan].parameters():
                            p.requires_grad = True
                    for p in whole.proto_list[chan].parameters():
                        p.requires_grad = True
                    whole.proto_list[chan].prototypes.requires_grad = True
                    if exp_var_dict['location_scale'] == True:
                        whole.proto_list[chan].location_scale.requires_grad = True
                for p in whole.final_list[0].last_layer.parameters():
                    p.requires_grad= False

            #Stage 3:
            if stage == 3:
                optimizer = stage3_optimizer
                for p in whole.base_list[0].parameters():
                    p.requires_grad = False
                for chan in range(exp_var_dict['num_channels']):
                    if exp_var_dict['sep_enc'] == True:
                        for p in whole.base_list[chan].parameters():
                            p.requires_grad = False
                    for p in whole.proto_list[chan].parameters():
                        p.requires_grad = False
                    whole.proto_list[chan].prototypes.requires_grad = False
                    if exp_var_dict['location_scale'] == True:
                        whole.proto_list[chan].location_scale.requires_grad = False
                for p in whole.final_list[0].last_layer.parameters():
                    p.requires_grad= True
            
            if stage == 1 and push_epoch(epoch, not exp_var_dict['delay_push'], exp_var_dict['delay_push_epoch']) == True:
                enable_dropout = False
            else:
                enable_dropout = True

            output, min_dist_list, scaled_sim_list = whole(inputs, enable_dropout)

            # compute loss
            cross_entropy = cross_loss(output, labels.to(torch.long))
            output = nn.functional.softmax(output) 
            predicted = torch.max(output.data, 1)
            n_examples += labels.size(0)
            n_correct += torch.eq(predicted[1], labels).sum().item()

            if stage != 3:
                max_dist = (whole.proto_list[0].prototype_shape[2] #num_proto?
                                * whole.proto_list[0].prototype_shape[1]
                                * whole.proto_list[0].prototype_shape[0])

                cluster_cost = 0
                separation_cost = 0
                div_loss = 0

                for chan in range(exp_var_dict['num_channels']):
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(whole.proto_list[chan].prototype_class_identity[:,labels.to(torch.int64)])
                    inverted_distances, _ = torch.max((max_dist - min_dist_list[chan]) * prototypes_of_correct_class, dim=1)
                    cluster_cost += torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_dist_list[chan]) * prototypes_of_wrong_class, dim=1)
                    separation_cost += torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                    div_loss += torch.sum(torch.clamp(exp_var_dict['div_threshold']-torch.cdist(torch.squeeze(whole.proto_list[chan].prototypes), torch.squeeze(whole.proto_list[chan].prototypes)), min=0)**2)
            
                epoch_cluster_cost += cluster_cost.item()
                epoch_separation_cost += separation_cost.item()
                #epoch_avg_separation_cost += avg_separation_cost.item()
                epoch_div_cost += div_loss.item()

            epoch_cross_entropy += cross_entropy.item()
            #regularization term 
            l1 = whole.final_list[0].last_layer.weight.norm(p=1)

        if stage !=3:
            loss = cross_entropy + exp_var_dict['cluster_cost_coef'] * cluster_cost - exp_var_dict['sep_cost_coef'] * separation_cost + exp_var_dict['div_cost_coef'] * div_loss 
        else:
            loss = cross_entropy + exp_var_dict['l1_coef'] * l1

        epoch_loss+= loss.item()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    scheduler.step()
    scheduler3.step()
    #print(scheduler.get_last_lr())
    #print(scheduler3.get_last_lr())

    print('Training')
    print('acc: '+str(n_correct/n_examples))
    print('Epoch Cross Entropy: '+str(epoch_cross_entropy))
    print('Epoch Cluster Cost: '+str(epoch_cluster_cost))
    print('Epoch Separation Cost: '+str(epoch_separation_cost))
    print('Epoch Div Cost: '+str(epoch_div_cost))

    #validation 
    epoch_cross_entropy = 0
    epoch_loss = 0
    n_examples = 0
    n_correct = 0

    for inputs, labels in val_dataloader:
        inputs = torch.nan_to_num(inputs)
        whole.base_list[0].eval()
        for chan in range(exp_var_dict['num_channels']):
            if exp_var_dict['sep_enc'] == True:
                whole.base_list[chan].eval()
            whole.proto_list[chan].eval()
        whole.final_list[0].eval()
        with torch.no_grad():
            output, min_dist_list, scaled_sim_list = whole(inputs, enable_dropout)

            cross_entropy = cross_loss(output, labels.to(torch.long))
            output = torch.nn.functional.softmax(output)
            predicted = torch.max(output.data, 1)
            n_examples += labels.size(0)
            n_correct += torch.eq(predicted[1], labels).sum().item()
            n_correct_1 += (abs(predicted[1]-labels) < 2).sum().item()

            if stage != 3:
                max_dist = (whole.proto_list[0].prototype_shape[2] #num_proto?
                                * whole.proto_list[0].prototype_shape[1]
                                * whole.proto_list[0].prototype_shape[0])

                cluster_cost = 0
                separation_cost = 0
                div_loss = 0

                for chan in range(exp_var_dict['num_channels']):
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(whole.proto_list[chan].prototype_class_identity[:,labels.to(torch.int64)])
                    inverted_distances, _ = torch.max((max_dist - min_dist_list[chan]) * prototypes_of_correct_class, dim=1)
                    cluster_cost += torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_dist_list[chan]) * prototypes_of_wrong_class, dim=1)
                    separation_cost += torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                    div_loss += torch.sum(torch.clamp(exp_var_dict['div_threshold']-torch.cdist(torch.squeeze(whole.proto_list[chan].prototypes), torch.squeeze(whole.proto_list[chan].prototypes)), min=0)**2)
            
                epoch_cluster_cost += cluster_cost.item()
                epoch_separation_cost += separation_cost.item()
                #epoch_avg_separation_cost += avg_separation_cost.item()
                epoch_div_cost += div_loss.item()
                #print(div_loss)

            epoch_cross_entropy += cross_entropy.item()
            #regularization term 
            l1 = whole.final_list[0].last_layer.weight.norm(p=1)
            
            if stage !=3:
                loss = cross_entropy + exp_var_dict['cluster_cost_coef'] * cluster_cost - exp_var_dict['sep_cost_coef'] * separation_cost + exp_var_dict['div_cost_coef'] * div_loss 
            else:
                loss = cross_entropy + exp_var_dict['l1_coef'] * l1

        epoch_loss+= loss.item()
        epoch_cross_entropy += cross_entropy.item()

    acc = n_correct/n_examples
    print('Validation Accuracy: '+str(acc))
    validation_acc_arr.append(acc)
    
    #Determine whether to go into Stage 2 projection or continue with Stage 3
    if stage == 1:
        if push_epoch(epoch, not exp_var_dict['delay_push'], exp_var_dict['delay_push_epoch']) == True:
            stage = 2
        else:
            stage = 3
    elif stage == 3:
        if epoch < train_epochs - (exp_var_dict['extra_three']+1): #Determines whether to run extra Stage 3 training after prototypes are learnt (only for MJO case study)
            stage = 1
        else:
            stage  = 3

    if stage == 2:
        print('Stage 2: starting push')
        prototype_info_list = []
        for train_images, train_labels in push_dataloader:
            prototypes_of_correct_class_train = torch.t(whole.proto_list[0].prototype_class_identity[:,train_labels.to(torch.long)])
            train_images = torch.nan_to_num(train_images)
            whole.base_list[0].eval()
            for chan in range(exp_var_dict['num_channels']):
                if exp_var_dict['sep_enc'] == True:
                    whole.base_list[chan].eval()
                whole.proto_list[chan].eval()
            whole.final_list[0].eval()
                
            with torch.no_grad():
                for chan in range(exp_var_dict['num_channels']):
                    if nc == False:
                        if exp_var_dict['sep_enc'] == True:
                            base_num, proto_num, prototype_info_num = push(whole.base_list[chan], whole.proto_list[chan], torch.unsqueeze(train_images[:,chan,:,:],1), prototypes_of_correct_class_train)
                        else:
                            base_num, proto_num, prototype_info_num = push(whole.base_list[0], whole.proto_list[chan], torch.unsqueeze(train_images[:,chan,:,:],1), prototypes_of_correct_class_train)
                    else:
                        base_num, proto_num, prototype_info_num = push(whole.base_list[chan], whole.proto_list[chan], train_images[:,:,:,:], prototypes_of_correct_class_train)

                    prototype_info_list.append(prototype_info_num)
        stage = 3
        print('push complete')


#to save trained model and prototypes
"""
if model_save == True:
    with open(prototype_info, "wb") as fp:   
        pickle.dump(prototype_info_list, fp)
    torch.save(whole.state_dict(), model_name)
    #raise Exception("end")
"""

#Testing 
epoch_cross_entropy = 0
n_examples = 0
n_correct = 0
n_correct_1 = 0 #+1/-1 accuracy
correct = []
true_label = []
pred_label = []

for inputs, labels in test_dataloader:
    inputs = torch.nan_to_num(inputs)
    whole.base_list[0].eval()
    for chan in range(exp_var_dict['num_channels']):
        if exp_var_dict['sep_enc'] == True:
            whole.base_list[chan].eval()
        whole.proto_list[chan].eval()
    whole.final_list[0].eval()
    with torch.no_grad():
        output, min_dist_list, scaled_sim_list = whole(inputs, enable_dropout)
        cross_entropy = cross_loss(output, labels.to(torch.long))
        output = torch.nn.functional.softmax(output)
        predicted = torch.max(output.data, 1)
        n_examples += labels.size(0)
        n_correct += torch.eq(predicted[1], labels).sum().item()
        n_correct_1 += (abs(predicted[1]-labels) < 2).sum().item()

        epoch_cross_entropy += cross_entropy.item()
        correct.append(torch.eq(predicted[1], labels).sum().item())
        true_label.append(int(labels.cpu().item()))
        pred_label.append(predicted[1].cpu().item())

acc = n_correct/n_examples
print('Testing accuracy: ' + str(acc))
print('+1 Testing accuracy: ' + str(n_correct_1/n_examples))

