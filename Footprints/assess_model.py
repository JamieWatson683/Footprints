import numpy as np
import matplotlib.pyplot as plt
import torch


def load_model(model_file):
    # Load model trained on GPU into CPU
    return torch.load(model_file, map_location=lambda storage, loc: storage)


def get_sample_heatmaps(model, samples):
    inputs = samples[:, 0:-1, :, :]
    labels = samples[:, -1, :, :]
    inputs = torch.from_numpy(inputs).float()
    with torch.no_grad():
        model.forward(inputs)
        heatmaps = model.final.probability
    return heatmaps.numpy(), labels


def load_samples(path, indices):
    samples = []
    for index in indices:
        sample = np.load(path+"data_"+str(index)+".npy")
        samples.append(sample)
    return np.array(samples)


def compare_heatmap(image, heatmap, label):
    plt.figure(figsize=(15,15))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(label, cmap='gray')
    plt.show()


