import numpy as np
import matplotlib.pyplot as plt
import torch


def load_model(model_file):
    # Load model trained on GPU into CPU
    return torch.load(model_file, map_location=lambda storage, loc: storage)

def assess_samples(samples):
    pass