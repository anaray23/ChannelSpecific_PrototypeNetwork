import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import h5py
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from scipy.ndimage import gaussian_filter as gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import random


def get_mnist_digit(digit, dataset):
    indices = [i for i, label in enumerate(dataset.targets) if label == digit]
    img, _ = dataset[random.choice(indices)]
    return img.squeeze(0).numpy()

def create_synthetic_sample(mnist_dataset):
    # random MNIST digit for all four quadrants (Channel 1)
    EPSILON = 1e-3
    base_digit = random.randint(0, 9)
    base_image = get_mnist_digit(base_digit, dataset=mnist_dataset)

    # choose even and an odd number
    even_digit = random.choice([0, 2, 4, 6, 8])
    odd_digit = random.choice([1, 3, 5, 7, 9])

    even = random.randint(0,1)
    if even == 0:
        even_label = True
    else:
        even_label = False


    # Create blank quadrants (28x28 blank images)
    blank = np.zeros((28, 28))


    if even_label:
        even_image = np.hstack([blank, get_mnist_digit(even_digit)])  # Left blank, right filled
        odd_image = np.hstack([blank, get_mnist_digit(odd_digit)])  
        label = even_digit
    else:
        even_image = np.hstack([get_mnist_digit(even_digit), blank])  # Left filled, right blank
        odd_image = np.hstack([get_mnist_digit(odd_digit), blank])  
        label = odd_digit

    # Stack vertically to form full-channel images
    channel_2 = np.vstack([even_image, even_image])  # Even digit appears only on one side
    channel_3 = np.vstack([odd_image, odd_image])  # Odd digit appears on the other side
    channel_1 = np.vstack([np.hstack([base_image, base_image])] * 2)  # Base digit appears in all quadrants

    channel_1 = channel_1 + EPSILON
    channel_2 = channel_2 + EPSILON
    channel_3 = channel_3 + EPSILON

    channel_1 = np.clip(channel_1, 0, 1)
    channel_2 = np.clip(channel_2, 0, 1)
    channel_3 = np.clip(channel_3, 0, 1)

    # final 3-channel image 
    synthetic_image = np.stack([channel_1, channel_2, channel_3], axis=0)

    return synthetic_image, channel_1, channel_2, channel_3, label

def visualize_synthetic_image(ch1, ch2, ch3, synthetic_image, label):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(ch1, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Channel 1: Base Digit")

    axes[1].imshow(ch2, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Channel 2: Even Digit")

    axes[2].imshow(ch3, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Channel 3: Odd Digit")

    axes[3].imshow(synthetic_image.sum(axis=0), cmap='gray', vmin=0, vmax=3)
    axes[3].set_title(f"Combined Image (Label: {label})")

    for ax in axes:
        ax.axis("off")

    plt.show()

def preprocess_mnist(file_name, val_split=0.2, normalization='default'):
    with h5py.File(file_name,'a') as f:
        syn_images = f["synthetic_images"][:]
        syn_labels = f["synthetic_labels"][:]
        print(syn_images.shape)
        print(syn_labels.shape)

        num_samples = len(syn_images)
        
        #shuffle data
        ind = np.arange(num_samples)
        np.random.shuffle(ind)
        print(ind.shape)
        images = syn_images[ind,:,:,:]
        labels = syn_labels[ind]

        print(images.shape)
        print(labels.shape)

        #train, val, test split
        test_split = 0.1
        test_ind = int(np.floor(test_split*num_samples))
        
        train_val_images = images[test_ind:]
        train_val_labels = labels[test_ind:]

        print(train_val_images.shape)

        test_images = images[:test_ind]
        test_labels = labels[:test_ind]

        print(test_images.shape)

        num_train_val = len(train_val_images)

        val_ind = int(np.floor(val_split*num_train_val))
        train_images = train_val_images[val_ind:]
        train_labels = train_val_labels[val_ind:]

        print(train_images.shape)

        val_images = train_val_images[:val_ind]
        val_labels = train_val_labels[:val_ind]

        print(val_images.shape)

        #normalize based on training data
        if normalization == 'all':
            image_mean  = np.mean(train_images)
            image_std   = np.std(train_images)
        elif normalization == 'pixel':
            image_mean  = np.mean(train_images,axis=(0,))
            image_std   = np.std(train_images,axis=(0,))
            #train_std[train_std==0] = 1
        else:
            image_mean = 0
            image_std = 1
        print(image_mean)
        print(image_std)
        train_images = (train_images-image_mean)/image_std
        val_images = (val_images-image_mean)/image_std
        test_images = (test_images-image_mean)/image_std
        if 'train_images' in f:
            del f['train_images']
            del f['train_labels']
            del f['val_images']
            del f['val_labels']
            del f['test_images']
            del f['test_labels']

    f.create_dataset("train_images", shape=train_images.shape, data = train_images)
    f.create_dataset("train_labels", shape=train_labels.shape, data = train_labels)

    f.create_dataset("val_images", shape=val_images.shape, data = val_images)
    f.create_dataset("val_labels", shape=val_labels.shape, data = val_labels)

    f.create_dataset("test_images", shape=test_images.shape, data = test_images)
    f.create_dataset("test_labels", shape=test_labels.shape, data = test_labels)

    print(f['train_images'].shape)
    print(f['val_images'].shape)
    print(f['test_images'].shape)


if __name__ == "__main__":
            
    # Load MNIST dataset 
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    num_samples = 12000
    synthetic_dataset = np.zeros((num_samples, 3, 56, 56), dtype=np.float32)
    clabels = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        img, _, _, _, lbl = create_synthetic_sample(mnist_dataset)
        synthetic_dataset[i] = img
        clabels[i] = lbl
        if i % 500 == 0:
            print(f"Generated {i}/{num_samples} images...")

    with h5py.File("data/syntheticMNIST/synthetic_data_channelmnist.hdf5",'a') as f:
        images = f.create_dataset("synthetic_images", shape=synthetic_dataset.shape, data=synthetic_dataset)
        labels = f.create_dataset("synthetic_labels", shape=clabels.shape, data=clabels)

    file_name = 'data/syntheticMNIST/synthetic_data_channelmnist.hdf5'
    preprocess_mnist(file_name)