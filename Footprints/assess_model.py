import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import datasets


def load_model(model_file):
    # Load model trained on GPU into CPU
    model = torch.load(model_file, map_location=lambda storage, loc: storage)
    return model


def get_sample_heatmaps(model, samples):
    inputs = samples[:, 0:-1, :, :]
    labels = samples[:, -1, :, :]
    inputs = torch.from_numpy(inputs).float()
    with torch.no_grad():
        model.eval()
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


def IoU(predictions, labels):
    intersection = np.multiply(predictions, labels).sum(-1).sum(-1)
    union = np.maximum(predictions, labels).sum(-1).sum(-1)
    return intersection / union


def get_distribution_of_ious(logs_path, model_name, dataloader):
    print("Loading model...")
    model = load_model(logs_path+model_name)
    model.eval()
    print("Success")
    with torch.no_grad():
        iou_list = []
        mask_sizes = []
        print("Getting IoU results...")
        for i, samples in enumerate(dataloader):
            print("Batch number {} of {}".format(i, len(dataloader)))
            inputs = samples['inputs'].float()
            labels = samples['labels'].float().unsqueeze(1)
            mask = inputs[:,-1,:,:]
            pixels = mask.sum(-1).sum(-1)
            pixels = torch.Tensor.tolist(pixels / (128*256))
            model.forward(inputs)
            predictions = model.final.probability > 0.5
            ious = torch.Tensor.tolist(IoU(predictions, labels))
            iou_list = iou_list + ious
            mask_sizes = mask_sizes + pixels
        iou_results = np.zeros((len(iou_list),2))
        iou_results[:,0] = np.squeeze(np.array(iou_list))
        iou_results[:,1] = np.squeeze(np.array(mask_sizes))
    return iou_results


if __name__=='__main__':
    dataset = datasets.FootprintsDataset("./data/validation_data/", augment=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    iou_results = get_distribution_of_ious(logs_path="./training_logs/RMSProp_no_occlusion/", model_name="model.pt",
                                           dataloader=dataloader)
    plt.figure()
    plt.hist(iou_results[:,0])
    plt.show()
