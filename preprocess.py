import os
import os.path
import numpy as np
from scipy.io import loadmat


def get_mat_data(path):
    """
    Gets all .mat files in the specified directory and subdirectories as a list of numpy arrays.
    NOTE: This is a *list* of numpy arrays, because the matrices retrieved from each image are not
    necessarily the same size.
    
    :param path: A string pointing to the top most directory
    :return: Returns (data, max_shape), where data is the array of matricies, and max_shape is the
        maximum detected shape in the search.
    """
    data = []
    max_shape = [0, 0]
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".mat")]:
            pathname = os.path.join(dirpath, filename)
            matrix = loadmat(pathname)['truth']
            max_shape[0] = max(max_shape[0], matrix.shape[0])
            max_shape[1] = max(max_shape[1], matrix.shape[1])
            # print("max_shape: ", max_shape, ", data shape: ", matrix.shape)
            data.append(matrix)
    return data, max_shape


def convert_and_pad(data, final_shape):
    """
    Converts a list of numpy matricies to a higher order numpy tensor, with padding
    
    :param data: The list of numpy matricies
    :param final_shape: A tuple representing the final shape of the result. 
        Typically should be the second return value from get_mat_data()
    """
    result = np.empty([len(data)] + final_shape)
    for idx, frame in enumerate(data):
        result[idx] = np.resize(frame, final_shape)
    return result
