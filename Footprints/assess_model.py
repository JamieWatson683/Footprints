import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import datasets
import os


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


def get_distribution_of_ious(logs_path, model_name, dataloader, use_GPU=False):
    print("Loading model...")
    model = load_model(logs_path+model_name)
    model.eval()
    print("Success")
    with torch.no_grad():
        if use_GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            gpu = torch.device('cuda:0')
            print("Using GPU device {}".format(gpu))
            model.cuda(device=gpu)
        iou_list = []
        mask_sizes = []
        print("Getting IoU results...")
        for i, samples in enumerate(dataloader):
            print("Batch number {} of {}".format(i, len(dataloader)))
            inputs = samples['inputs'].float()
            labels = samples['labels'].float().unsqueeze(1)
            if use_GPU:
                    inputs = inputs.to(device=gpu)
                    labels = labels.to(device=gpu)
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
    # dataset = datasets.FootprintsDataset("./data/validation_data/", augment=False)
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # iou_results = get_distribution_of_ious(logs_path="./training_logs/RMSProp_no_occlusion/", model_name="model.pt",
    #                                        dataloader=dataloader, use_GPU=True)
    # np.save("./training_logs/RMSProp_no_occlusion/iou_results.npy", iou_results)

    data = np.load("./training_logs/RMSProp_no_occlusion/iou_results.npy")
    small = data[:,0][data[:,1]<=0.025]
    moderate = data[:,0][np.logical_and(data[:,1]<=0.05, data[:,1]>0.025)]
    large = data[:,0][data[:,1]>0.05]
    plt.figure()
    plt.hist(small, bins=50)
    plt.show()
    plt.figure()
    plt.hist(moderate, bins=50)
    plt.show()
    plt.figure()
    plt.hist(large, bins=50)
    plt.show()

