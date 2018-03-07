import numpy as np

def split_data(data, proportion): #function taken from hackathon_3 notebook and improved by Jingchao Zhang
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1

    Returns: 
        - train: numpy array; used for training
        - val: numpy array; used for validation
    """
    size = data.shape[0]
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    train = data[s[:split_idx]]
    val = data[s[split_idx:]]
    return train, val

def k_fold_split(data, k, i): #function improved upon Paul Quint's draft
    """
    Partitions a numpy array into 'training set' and 'validation set' for k-fold validation

    Args:
        - data: numpy array; to be aplit along the first axis
        - k: positive integer; the data will be evenly split into k groups. 
        - i: integer; 0 <= i < k; the i-th group is used as the validation set, and the rest is used as 
        the training set. 

    Returns: 
        - train: numpy array; used for training
        - val: numpy array; used for validation
    """
    t = data.shape[0]//k
    val = data[i*t:(i+1)*t]
    first = data[:i*t]
    second = data[(i+1)*t:]
    train = np.concatenate((first, second), axis = 0)
    return train, val