

import torch.nn as nn
import torch



exp_var_dict = {
    'exp_dataset' : 'mnist',
    'img_input_channels' : 1, 
    'num_prototypes' : 50,
    'prototype_shape': [32,1,1],
    'img_size': [2,2],
    'proto_img_shape' : [2,2],
    'num_proto_per_class':5,
    'num_classes' : 10,
    'batch_size' : 32,
    'location_scale' : True,
    'base_hidden_layer_input_size' : 128,
    'base_hidden_layer_output_size' : 32,
    'proto_input_channels' : 8,
    'proto_input_channels1' : 16,
    'proto_input_channels2' : 32,
    'proto_output_channels' :32,
    'cluster_cost_coef': 0.7,
    'sep_cost_coef': 0.7, 
    'l1_coef': 1e-2,
    'div_cost_coef':0.001,
    'div_threshold':0.001, 
    'random_seed':'nan',
    'num_channels':3,
    'neg_strength':False,
    'pos_strength':0.001,
    'stage1_lr':0.001, 
    'stage3_lr':0.001,
    'delay_push': True,
    'delay_push_epoch':6,
    'inactive_threshold':0.5,
    'extra_three':0,
    'sep_enc':False,
}




"""
exp_var_dict = {
    'exp_dataset' : 'mjo',
    'img_input_channels' : 1, 
    'num_prototypes' : 90, 
    'prototype_shape': [64,1,1],
    'img_size': [16,131],
    'proto_img_shape' : [2,5],
    'num_proto_per_class':10,
    'num_classes' : 9,
    'batch_size' : 32,
    'location_scale' : True,
    'base_hidden_layer_input_size' : 640, 
    'base_hidden_layer_output_size' : 32, 
    'proto_input_channels' : 16,
    'proto_input_channels1' : 32,
    'proto_input_channels2' : 64,
    'proto_output_channels' :64,
    'cluster_cost_coef': 0.5,
    'sep_cost_coef': 0.2,
    'l1_coef': 1e-3,
    'div_cost_coef':0.01,
    'div_threshold':0.001,
    'random_seed':'nan',
    'num_channels':3, 
    'neg_strength':False, 
    'pos_strength':0.2,
    'stage1_lr':0.001,
    'stage3_lr':0.001,
    'delay_push': True,
    'delay_push_epoch':6,
    'inactive_threshold':0.5,
    'extra_three':15,
    'sep_enc':False,
}
"""

#MJO with noise
"""
exp_var_dict = {
    'exp_dataset' : 'mjo',
    'img_input_channels' : 1, 
    'num_prototypes' : 90, 
    'prototype_shape': [64,1,1],
    'img_size': [16,131],
    'proto_img_shape' : [2,5],
    'num_proto_per_class':10, 
    'num_classes' : 9,
    'batch_size' : 32,
    'location_scale' : True,
    'base_hidden_layer_input_size' : 640, 
    'base_hidden_layer_output_size' : 32, 
    'proto_input_channels' : 16,
    'proto_input_channels1' : 32,
    'proto_input_channels2' : 64,
    'proto_output_channels' :64,
    'cluster_cost_coef': 0.5,
    'sep_cost_coef': 0.2,
    'l1_coef': 1e-3,
    'div_cost_coef':0.01,
    'div_threshold':0.001,
    'random_seed':'nan',
    'num_channels':4, 
    'neg_strength':False, 
    'pos_strength':0.2,
    'stage1_lr':0.001,
    'stage3_lr':0.001,
    'delay_push': True,
    'delay_push_epoch':6,
    'inactive_threshold':0.5,
    'extra_three':15,
    'sep_enc':False,
}
"""


"""
exp_var_dict = {
    'exp_dataset' : 'euro_rs',
    'img_input_channels' : 1, 
    'num_prototypes' : 40, 
    'prototype_shape': [64,1,1],
    'img_size': [64,64],
    'proto_img_shape' : [2,2],
    'num_proto_per_class':4, 
    'num_classes' : 10,
    'batch_size' : 64,
    'location_scale' : False,
    'base_hidden_layer_input_size' : 256,
    'base_hidden_layer_output_size' : 128, 
    'proto_input_channels' : 16,
    'proto_input_channels1' : 32,
    'proto_input_channels2' : 64,
    'proto_output_channels' :64,
    'cluster_cost_coef': 0.2,
    'sep_cost_coef': 0.02,
    'l1_coef': 1e-3,
    'div_cost_coef':0.01, 
    'div_threshold':0.1,
    'random_seed':'nan',
    'num_channels':13, 
    'neg_strength':False, 
    'pos_strength':0.2,
    'stage1_lr':0.001,
    'stage3_lr':0.001,
    'delay_push': True,
    'delay_push_epoch':4,
    'inactive_threshold':0.5,
    'extra_three':0,
    'sep_enc':False,
}

"""


class BaseConvModel(nn.Module):
    def __init__(self, exp, CNN_only=False):
        super().__init__()
        self.CNN_only = CNN_only
        self.exp = exp

        if self.exp == 'mjo':
            pool_stride = (2,3) 
        elif self.exp == 'mnist':
            pool_stride=(3,3)
        else:
            pool_stride = (2,2)

        #Base CNN
        if self.CNN_only == True:
            self.conv1 = nn.Conv2d(exp_var_dict['num_channels'], exp_var_dict['proto_input_channels'], kernel_size=(3,3), stride=1, padding='same') 
        else:
            self.conv1 = nn.Conv2d(1, exp_var_dict['proto_input_channels'], kernel_size=(3,3), stride=1, padding='same')
        self.act1 = nn.LeakyReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=pool_stride)

        self.conv2 = nn.Conv2d(exp_var_dict['proto_input_channels'], exp_var_dict['proto_input_channels1'], kernel_size=(3,3), stride=1, padding='same')
        self.act2 = nn.LeakyReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=(2,2), stride=pool_stride)
        if self.CNN_only == True:
            self.drop = nn.Dropout(0.5)
        else:
            self.drop = nn.Dropout(0.2) 

        self.conv3 = nn.Conv2d(exp_var_dict['proto_input_channels1'], exp_var_dict['proto_input_channels2'], kernel_size=(3,3), stride=1, padding='same')
        self.act3 = nn.LeakyReLU()

        if self.exp == 'euro_rs':
            self.pool3 = nn.AvgPool2d(kernel_size=(2,2), stride=pool_stride)
        elif self.exp == 'mnist':
            self.pool3 = nn.AvgPool2d(kernel_size=(3,3), stride=pool_stride)
        elif self.exp == 'mjo':
            self.pool3 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,3)) 
        self.drop2 = nn.Dropout(0.2) 

        if self.exp ==  'euro_rs':
            self.conv4 = nn.Conv2d(exp_var_dict['proto_input_channels2'], exp_var_dict['proto_input_channels2'], kernel_size=(3,3), stride=1, padding='same')
            self.act4 = nn.LeakyReLU()
            self.pool4 = nn.AvgPool2d(kernel_size=(2,2), stride=pool_stride)
            self.pool5 = nn.AvgPool2d(kernel_size=(2,2), stride=pool_stride)
        elif self.exp == 'mjo':
            self.conv4 = nn.Conv2d(exp_var_dict['proto_input_channels2'], exp_var_dict['proto_input_channels2'], kernel_size=(3,3), stride=1, padding='same')
            self.act4 = nn.LeakyReLU()

        
        if self.CNN_only==True:
            self.flatten = nn.Flatten()
            
            if self.exp == 'mjo':
                self.lin1 = nn.Linear(exp_var_dict['base_hidden_layer_input_size'], exp_var_dict['base_hidden_layer_output_size'])
                self.act5 = nn.LeakyReLU()
                self.lin2 = nn.Linear(exp_var_dict['base_hidden_layer_output_size'], exp_var_dict['num_classes'])
                self.base_layers = nn.ModuleList([self.conv1, self.act1, self.pool1, self.conv2, self.act2, self.pool2, self.conv3, self.act3, self.pool3, self.conv4, self.act4, self.lin1, self.lin2])
            elif self.exp ==  'euro_rs':
                self.lin1 = nn.Linear(exp_var_dict['base_hidden_layer_input_size'], exp_var_dict['base_hidden_layer_output_size'])
                self.act5 = nn.LeakyReLU()
                self.lin2 = nn.Linear(exp_var_dict['base_hidden_layer_output_size'], exp_var_dict['num_classes'])
                self.base_layers = nn.ModuleList([self.conv1, self.act1, self.pool1, self.conv2, self.act2, self.pool2, self.conv3, self.act3, self.pool3, self.conv4, self.act4, self.pool4, self.pool5, self.lin1, self.lin2])
            else:
                self.lin1 = nn.Linear(exp_var_dict['base_hidden_layer_input_size'], exp_var_dict['num_classes'])
                self.base_layers = nn.ModuleList([self.conv1, self.act1, self.pool1, self.conv2, self.act2, self.pool2, self.conv3, self.act3, self.pool3, self.lin1])
        
        elif self.exp ==  'euro_rs':
            self.base_layers = nn.ModuleList([self.conv1, self.act1, self.pool1, self.conv2, self.act2, self.pool2, self.conv3, self.act3, self.pool3, self.conv4, self.act4, self.pool4, self.pool5]) 
        elif self.exp == 'mjo':
            self.base_layers = nn.ModuleList([self.conv1, self.act1, self.pool1, self.conv2,  self.act2, self.pool2, self.conv3, self.act3, self.pool3, self.conv4, self.act4]) 
        else:
            self.base_layers = nn.ModuleList([self.conv1, self.act1, self.pool1, self.conv2, self.act2, self.pool2, self.conv3, self.act3, self.pool3]) 


    def forward(self, x, enable_dropout=True):
        x = self.conv1(x)
        x = self.pool1(self.act1(x))
        
        x = self.conv2(x)
        x = self.pool2(self.act2(x))
        if self.exp == 'mjo' and enable_dropout==True:
            x = self.drop(x)

        x = self.conv3(x)
        x = self.pool3(self.act3(x))
        if enable_dropout==True:
            x = self.drop2(x)
        
        if self.exp == 'euro_rs':
            x = self.conv4(x)
            x = self.pool4(self.act4(x))
            x = self.pool5(x)
        elif self.exp == 'mjo':
            x = self.conv4(x)
            x = self.act4(x)
        if self.CNN_only == True:
            x = self.flatten(x)
            if self.exp == 'mjo':
                x = self.act5(self.lin1(x))
                x = self.lin2(x)
            elif self.exp == 'euro_rs':
                x = self.act5(self.lin1(x))
                x = self.lin2(x)
            else:
                x = self.lin1(x)
        return x

#class modified from Barnes EA et. al (2022) doi: 10.1175/AIES-D-22-0001.1 and Chen C et al (2019) doi: 10.48550/arXiv.1806.10574 to include channel-specific prototypes
class ProtoModel(nn.Module):
    def __init__(self, img_size, num_prototypes, prototype_shape, num_classes):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.prototype_shape = prototype_shape
        self.img_size = img_size
        self.num_classes = num_classes
        self.epsilon = torch.ones([1])*0.0001

        # 1x1 CNN Layers
        self.conv3 = nn.Conv2d(exp_var_dict['proto_input_channels2'], exp_var_dict['proto_output_channels'], kernel_size=(1,1)) 
        self.act3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(exp_var_dict['proto_output_channels'],exp_var_dict['proto_output_channels'], kernel_size=(1,1)) 
        self.act4 = nn.LeakyReLU() 

        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1


        self.prototypes = nn.Parameter(torch.rand(self.num_prototypes, *self.prototype_shape), requires_grad = True)
        if exp_var_dict['location_scale'] == True:
            self.location_scale = nn.Parameter(torch.zeros(self.num_prototypes, *img_size), requires_grad = True)
        else:
            self.location_scale = torch.zeros(self.num_prototypes, *img_size)

        self.proto_layers = nn.ModuleList([self.conv3, self.act3, self.conv4, self.act4])

    def l2_distances(self, x):

        xTx = nn.functional.conv2d(input=x**2, weight=torch.ones(self.num_prototypes, *self.prototype_shape)) 
        xTy = nn.functional.conv2d(input=x, weight=self.prototypes)
        yTy = torch.sum(self.prototypes**2, dim=(1)) 

        distances = nn.functional.relu(xTx - 2*xTy + yTy)

        return distances
    
    def distance_2_similarity(self, distances, location_scale_factor):
        similarity_scores = torch.log((distances + 1) / (distances + self.epsilon)) 
        scaled_similarity_scores = torch.multiply(similarity_scores, location_scale_factor) 
        return torch.amax(scaled_similarity_scores, dim=(2,3)), scaled_similarity_scores 
        

    def forward(self, x):
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))

        location_scale_factor = torch.exp(self.location_scale)
        distances = self.l2_distances(x)

        min_distances = torch.amin(distances / (location_scale_factor + self.epsilon), dim=(2,3)) 
        max_similarity_score, scaled_similarity_scores = self.distance_2_similarity(distances, location_scale_factor)

        return max_similarity_score, min_distances, scaled_similarity_scores 

    def push_forward(self, x):
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        location_scale_factor = torch.exp(self.location_scale)
        distances = self.l2_distances(x)
        max_similarity_score, scaled_similarity_scores = self.distance_2_similarity(distances, location_scale_factor)

        return x, scaled_similarity_scores


class FinalLayer(nn.Module):
    def __init__(self, num_classes, num_prototypes, neg_strength, pos_strength):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_classes = num_classes

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)
        if neg_strength != False:
            self.neg_strength = neg_strength
            self.pos_strength = pos_strength
        
            self.weights_init = torch.full((exp_var_dict['num_prototypes'], exp_var_dict['num_classes']),self.neg_strength)

            num_prototypes_per_class = exp_var_dict['num_proto_per_class']
            for j in range(exp_var_dict['num_prototypes']):
                self.weights_init[j, j // num_prototypes_per_class] = self.pos_strength

            self.weights_init = self.weights_init.repeat(exp_var_dict['num_channels'],1)
            self.last_layer.weight.data = self.weights_init.T
        
        self.final_layers = nn.ModuleList([self.last_layer])
        
    def forward(self, x):
        x = self.last_layer(x)
        return x

class WholeModel(nn.Module):
    def __init__(self, exp, num_channels, backbone_weights=False):
        super().__init__()
        self.num_channels = num_channels
        if backbone_weights== False:
            if exp_var_dict['sep_enc']== True:
                self.base_list = nn.ModuleList([BaseConvModel(exp_var_dict['exp_dataset']) for i in range(num_channels)])
            else:
                self.base_list = nn.ModuleList([BaseConvModel(exp_var_dict['exp_dataset'])])
        else:
            base_model = BaseConvModel(exp_var_dict['exp_dataset'])
            base_model.load_state_dict(backbone_weights, strict=False)
            self.base_list = nn.ModuleList([base_model])
        self.proto_list = nn.ModuleList([ProtoModel(img_size=exp_var_dict['proto_img_shape'], num_prototypes=exp_var_dict['num_prototypes'], prototype_shape=exp_var_dict['prototype_shape'], num_classes=exp_var_dict['num_classes']) for i in range(num_channels)])
        self.final_list = nn.ModuleList([FinalLayer(num_classes=exp_var_dict['num_classes'],num_prototypes=exp_var_dict['num_prototypes']*exp_var_dict['num_channels'], neg_strength=exp_var_dict['neg_strength'], pos_strength=exp_var_dict['pos_strength'])])
    
    def forward(self, x, enable_dropout=True):
        
        output_list = []
        min_dist_list = []
        scaled_sim_list = []
        for chan in range(self.num_channels):
            if exp_var_dict['sep_enc']== True:
                output_chan = self.base_list[chan](torch.unsqueeze(x[:,chan,:,:],1), enable_dropout) 
            else:
                output_chan = self.base_list[0](torch.unsqueeze(x[:,chan,:,:],1), enable_dropout) 

            output_chan, min_dist_chan, scaled_sim_chan = self.proto_list[chan](output_chan)
            output_list.append(output_chan)
            min_dist_list.append(min_dist_chan)
            scaled_sim_list.append(scaled_sim_chan)

        output = self.final_list[0](torch.concat(output_list,1))

        return output, min_dist_list, scaled_sim_list



    

