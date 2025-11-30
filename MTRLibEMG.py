# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:20:52 2024

@author: msi
"""
import numpy as np
def Znorm(feattrain,feattest):
    import numpy as np
    for i in feattrain:
        meanI = np.mean(feattrain[i], axis=0)
        sdI = np.std(feattrain[i], axis=0,ddof=1)
        sdI[sdI == 0] = 1
        feattrain[i] = (feattrain[i] - meanI) / sdI
        feattest[i] = (feattest[i] - meanI) / sdI
    
    return feattrain,feattest
def RawDSNORSPL(ws,si,trialno,gaitphase,fsemg,a,b):
    import numpy as np
    from libemg.utils import get_windows
    test_indices1 = np.where(trialno == a)[0]
    test_indices2 = np.where(trialno == b)[0]
    test_indices = np.concatenate([test_indices1, test_indices2])
    tsfsemg = fsemg[test_indices,:]
    gaitphase=gaitphase.astype(int)
    classtest= gaitphase[test_indices]
    logical_index = np.ones(len(trialno), dtype=bool)
    logical_index[test_indices] = False
    trfsemg=fsemg[logical_index,:]
    classtrain=gaitphase[logical_index]
    s=trfsemg.shape[1]
    for z in range(s):
     meanI = np.mean(trfsemg[:,z], axis=0)
     sdI = np.std(trfsemg[:,z], axis=0)
     trfsemg[:,z] = (trfsemg[:,z] - meanI) / sdI
     tsfsemg[:,z] = (tsfsemg[:,z] - meanI) / sdI
    data_train=get_windows(trfsemg,ws,si)
    data_test=get_windows(tsfsemg,ws,si)
    l1=data_train.shape[0]
    classtrain = classtrain[::si]
    l2 = classtrain.shape[0]
    l = min(l1, l2)
    classtrain = classtrain[:l]
    data_train = data_train[:l,:,:]
    l1=data_test.shape[0]
    classtest = classtest[::si]
    l2 = classtest.shape[0]
    l = min(l1, l2)
    classtest = classtest[:l]
    data_test = data_test[:l,:,:]
    return data_train,classtrain,data_test,classtest

def DSOut(si,trialno,gaitphase,emgfeat):
    
    first_key = next(iter(emgfeat))
    l1 = emgfeat[first_key].shape[0]
    gaitphase_downsampled = gaitphase[::si]
    if trialno is None:
        # If trialno is None, create a new trialno array assuming 10 trials of equal length
        num_trials = 10
        trial_length = l1 // num_trials
        trialno = np.concatenate([np.full(trial_length, i + 1) for i in range(num_trials)])
        # Handle any remaining samples due to uneven division
        trialno_downsampled=trialno
        if len(trialno) < l1:
            trialno_downsampled = np.concatenate([trialno, np.full(l1 - len(trialno), num_trials)])
            
    else:
        trialno_downsampled=trialno[::si]
    l2 = trialno_downsampled.shape[0]
    l = min(l1, l2)
    for i in emgfeat:
        emgfeat[i]=emgfeat[i][:l,:]
    gaitphase_downsampled = gaitphase_downsampled[:l]
    trialno_downsampled = trialno_downsampled[:l]
    return emgfeat,gaitphase_downsampled,trialno_downsampled

def RawDSOut(si,trialno,gaitphase,windows):
    l1=windows.shape[0]
    gaitphase_downsampled = gaitphase[::si]
    trialno_downsampled = trialno[::si]
    l2 = trialno_downsampled.shape[0]
    l = min(l1, l2)
    windows=windows[:l,:,:]
    gaitphase_downsampled = gaitphase_downsampled[:l]
    trialno_downsampled = trialno_downsampled[:l]
    return windows,gaitphase_downsampled,trialno_downsampled

def DataSplit(emgfeat,trialno_downsampled,gaitphase_downsampled,a,b=None):

    import numpy as np
    test_indices1 = np.where(trialno_downsampled == a)[0]
    if b is not None:
     test_indices2 = np.where(trialno_downsampled == b)[0]
     test_indices = np.concatenate([test_indices1, test_indices2])
    elif b is None:
     test_indices=test_indices1   
    feattest = {key: value[test_indices,:] for key, value in emgfeat.items()}
    classtest = gaitphase_downsampled[test_indices]
    logical_index = np.ones(len(trialno_downsampled), dtype=bool)
    logical_index[test_indices] = False
    feattrain =  {key: value[logical_index,:] for key, value in emgfeat.items()}
    classtrain = gaitphase_downsampled[logical_index]
    return feattrain, classtrain, feattest, classtest

def RawDataSplit(windows,trialno_downsampled,gaitphase_downsampled,a,b):

    import numpy as np
    test_indices1 = np.where(trialno_downsampled == a)[0]
    test_indices2 = np.where(trialno_downsampled == b)[0]
    test_indices = np.concatenate([test_indices1, test_indices2])
    data_test = windows[test_indices,:,:]
    classtest = gaitphase_downsampled[test_indices]
    logical_index = np.ones(len(trialno_downsampled), dtype=bool)
    logical_index[test_indices] = False
    data_train =  windows[logical_index,:,:]
    classtrain = gaitphase_downsampled[logical_index]
    return data_train, classtrain, data_test, classtest


def concatshiftvers(A, num_shifts):
    """
    Concatenate shifted versions of an array A.
    
    Parameters:
    A (np.ndarray): Input array of shape (m, n)
    num_shifts (int): Number of additional shifted versions to concatenate
    
    Returns:
    np.ndarray: Concatenated array with shifted versions
    """
    m, n = A.shape
    # Initialize the result array
    result = A[:m-num_shifts]
    
    # Concatenate shifted versions
    for shift in range(1, num_shifts + 1):
        result = np.hstack((result, A[shift:m-num_shifts+shift]))
    
    return result

def apply_shifted_concatenation_to_dict(train_dict,train_label,test_dict,test_label, num_shifts):
    """
    Apply concatenate_shifted_versions to all arrays in a dictionary.
    
    Parameters:
    data_dict (dict): Dictionary where each value is a numpy array
    num_shifts (int): Number of additional shifted versions to concatenate
    
    Returns:
    dict: New dictionary with concatenated shifted arrays
    """
    train = {}
    for key, array in train_dict.items():
        if isinstance(array, np.ndarray) and len(array.shape) == 2:
            train[key] = concatshiftvers(array, num_shifts)
        else:
            raise ValueError(f"Value for key '{key}' is not a 2D numpy array")
    train_label=train_label[0+num_shifts::]        
    test = {}
    for key, array in test_dict.items():
        if isinstance(array, np.ndarray) and len(array.shape) == 2:
            test[key] = concatshiftvers(array, num_shifts)
        else:
            raise ValueError(f"Value for key '{key}' is not a 2D numpy array")
    test_label=test_label[0+num_shifts::]                
    return train,train_label,test,test_label

def concsingleshift(A, r):
    """
    Concatenate a single shifted version of an array A by r samples.
    
    Parameters:
    A (np.ndarray): Input array of shape (m, n)
    r (int): index of samples to shift
    
    Returns:
    np.ndarray: Concatenated array with a single shifted version
    """
    m, n = A.shape
    if r >= m:
        raise ValueError("r must be less than the number of rows in A")
    
    result = np.hstack((A[:m-r], A[r:]))
    return result

def apply_single_indexed_shift(train_dict,train_label,test_dict,test_label, r):
    """
    Apply concatenate_shifted_versions to all arrays in a dictionary.
    
    Parameters:
    data_dict (dict): Dictionary where each value is a numpy array
    r (int): index of samples to shift
    
    Returns:
    dict: New dictionary with concatenated shifted arrays
    """
    train = {}
    for key, array in train_dict.items():
        if isinstance(array, np.ndarray) and len(array.shape) == 2:
            train[key] = concsingleshift(array, r)
        else:
            raise ValueError(f"Value for key '{key}' is not a 2D numpy array")
    train_label=train_label[0+r::]        
    test = {}
    for key, array in test_dict.items():
        if isinstance(array, np.ndarray) and len(array.shape) == 2:
            test[key] = concsingleshift(array, r)
        else:
            raise ValueError(f"Value for key '{key}' is not a 2D numpy array")
    test_label=test_label[0+r::]                
    return train,train_label,test,test_label