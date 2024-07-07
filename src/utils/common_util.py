import torch
import numpy as np
import sklearn.metrics as metrics
device = torch.device("cuda:0")

def preprocess(path):
    #  dictionary with different lentgth (5~13)
    total_tracks = np.load(path, allow_pickle=True).item()
    true_tracklets = torch.zeros((1, 5, 5)).to(device)
    false_tracklets = torch.zeros((1, 5, 5)).to(device)
    for i in range(5, 13):
        tracks = torch.from_numpy(total_tracks[str(i)]).to(device).float()
        track_num, track_length, _ = tracks.shape
        j = 0
        while j < (track_num - 1):
            for k in range(5, track_length):
                tracklet = tracks[j : j + 1, k - 5 : k, :]
                false_tracklet = torch.cat((
                            tracks[j : j + 1, k - 5 : k - 1, :],
                            tracks[j + 1 : j + 2, k - 1 : k, :],
                        ),axis=1)

                tracklet[:,0,2:4] = 0
                false_tracklet[:,0,2:4] = 0
                
                true_tracklets = torch.cat((true_tracklets, tracklet), axis=0)
                false_tracklets = torch.cat((false_tracklets, false_tracklet), axis=0)
            
            j = j + 2
    tracklets = torch.cat((true_tracklets[1:,:],false_tracklets[1:,:]),axis=0)
    
    input_seq = tracklets[:, :-1, :-1]
    candidates = tracklets[:, -2:, :-1]
    reg_label = torch.cat((true_tracklets[1:,-1,:],true_tracklets[1:,-1,:]),axis=0)[:,:-1]
    cls_label = tracklets[:, -1, -1].long()

    return input_seq, candidates, reg_label, cls_label

# normalize 
# Only x and Z are regularized, and X offset and Y offset are not regularized
def data_normal(origin_data,physical_width):
    data_min = 0
    data_max = physical_width
    normal_data = (origin_data - data_min)/(data_max - data_min)
    return normal_data

def inverse_data_normal(normal_data,physical_width):
    data_min = 0
    data_max = physical_width
    origin_data = normal_data*(data_max - data_min) + data_min
    return origin_data


# get confusion matrix
def confusion_matrix(y_true, y_pred,TP,FP,TN,FN):
    tp, fp, tn, fn = 0, 0, 0, 0
    y_pred = torch.argmax(y_pred,dim=1)
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
           tp += 1
        if y_true[i] == 0 and y_pred[i] == 1:
           fp += 1
        if y_true[i] == 0 and y_pred[i] == 0:
           tn += 1
        if y_true[i] == 1 and y_pred[i] == 0:
           fn += 1
    TP = tp+TP
    FP = fp+FP
    TN = tn+TN
    FN = fn+FN
    return TP, FP, TN, FN


