import numpy as np
import random

# change point data domain 
def convert(tracks):
    nums, length, _ = tracks.shape
    result = np.zeros((nums, length, 3))
    # x and z
    result[:, :, :2] = tracks[:, :, :2]
    result[:, :, -1:] = tracks[:, :, -1:]
    return result

#  divide total tracks into train, validate and test
def divide(tracks,train_num,val_num,test_num):
    # tracks for test
    test_set = tracks[-test_num*2:, :, :]
    
    # tracks for train+validate
    data = tracks[:-test_num*2, :, :]
    total_indexs = [x for x in range(train_num+val_num)]
    
    # tracks for train
    train_indexs = random.sample(range(0, train_num+val_num), train_num)
    train_indexs.sort()
    for j in range(len(train_indexs)):
        index = train_indexs[j]
        if j == 0:
            train_set = data[2*index:2*index+2, :, :]
            continue
        train_set = np.concatenate(
            (train_set, data[2*index:2*index+2, :, :]), axis=0)
    
    # tracks for validate
    val_indexs = [y for y in total_indexs if y not in train_indexs]
    val_indexs.sort()
    for k in range(len(val_indexs)):
        index = val_indexs[k]
        if k == 0:
            val_set = data[2*index:2*index+2, :, :]
            continue
        val_set = np.concatenate(
            (val_set, data[2*index:2*index+2, :, :]), axis=0)

    return convert(train_set),convert(val_set),convert(test_set)
