#script modified from Chen C et al (2019) doi: 10.48550/arXiv.1806.10574 to include channel-specific prototypes

import torch
import numpy as np

def push(base, proto, input_images, prototypes_of_correct_class_train):
    prototypes = proto.prototypes
    num_prototypes = proto.num_prototypes
    location_scale = proto.location_scale
    location_scale_factor = torch.exp(location_scale)


    new_prototypes = np.zeros(np.shape(prototypes))
    new_prototypes_indices = np.zeros((num_prototypes,2))

    output = base(input_images)
    conv_output, similarity_scores = proto.push_forward(output)

    new_prototype_sample = np.zeros((num_prototypes),dtype=np.float32)
    new_prototype_sample_sim = np.zeros((num_prototypes))
    
    similarity_scores = similarity_scores.cpu().numpy()
    prototypes_of_correct_class_train = prototypes_of_correct_class_train.cpu().numpy()
    old_proto_indices = []

    for prototype_index in range(0, num_prototypes):
        m = np.max(similarity_scores[:,prototype_index,:,:], axis=(1,2) ) 
        m = m*prototypes_of_correct_class_train[:,prototype_index]
        new_prototype_sample[prototype_index] = int(np.argmax(m))
        new_prototype_sample_sim[prototype_index] = np.max(m)

        x = similarity_scores[int(np.argmax(m)), prototype_index, :,:]
        j,k = np.unravel_index(np.argmax(x), shape=x.shape)
        push_prototype = conv_output[int(np.argmax(m)), :, j, k]
        old_proto_indices.append(int(np.argmax(m)))
        new_prototypes[prototype_index,:,0,0] = push_prototype.cpu()

        new_prototypes_indices[prototype_index,0] = j
        new_prototypes_indices[prototype_index,1] = k

        previous_prototype = prototypes[prototype_index,:,0,0]

    dist = np.mean(((proto.prototypes.data.cpu() - new_prototypes[:,:,:,:])).detach().numpy())
    dist = np.mean(((proto.prototypes.data.cpu() - new_prototypes[:,:,:,:])/proto.prototypes.data.cpu()).detach().numpy())

    proto.prototypes.data = torch.Tensor(new_prototypes[:,:,:,:])
    return base, proto, (new_prototype_sample, new_prototype_sample_sim, new_prototypes, similarity_scores, new_prototypes_indices, location_scale, input_images.shape, conv_output.shape)




