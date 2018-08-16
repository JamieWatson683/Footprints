import numpy as np
import os


def split_array(array_path, save_path):
    array = np.load(array_path)
    print("Splitting file {}".format(array_path))
    for i in range(len(array)):
        print(i)
        subarray = array[i,:,:,:]
        np.save(save_path + "data_" + str(i) + ".npy", subarray)
    print("Success")

def transpose_all(path):
    files = os.listdir(path)
    for file in files:
        array = np.load(path + file)
        array = np.transpose(array, [2,0,1])
        np.save(path + file, array)


if __name__ == '__main__':
    # split_array("./data/synthetic/training_data/arr_0.npy", "./data/synthetic/training_data/")
    # split_array("./data/synthetic/validation_data/arr_0.npy", "./data/synthetic/validation_data/")

    transpose_all("./data/synthetic/validation_data/")
    transpose_all("./data/synthetic/training_data/")
