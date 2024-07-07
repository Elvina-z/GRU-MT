import numpy as np
import cv2 as cv
import torch
import os
import sklearn.metrics as metrics
import torch.nn.functional as F
from utils.common_util import data_normal,inverse_data_normal
from net.net_gru import NetGRU as Net
from utils.infer_util import rmse_histogram,classify_histogram


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def RMSE(pred_dictionary, total_tracks):
    track_num, _, _ = pred_dictionary['5'].shape
    rmse_list = []
    for i in range(5,13):
        pred_tracks = pred_dictionary[str(i)].detach().numpy()
        true_tracks = total_tracks[str(i)]
        rmse = 0
        for k in range(track_num):
            rmse += np.sqrt(metrics.mean_squared_error(
                true_tracks[k, -1:, :-1], pred_tracks[k, -1:, :]))
        rmse_list.append(rmse/track_num)
    return rmse_list

def model_infer(model,total_tracks,physical_width):

    regression_rmse = []
    regression_mae = []
    classify_accuracy=[]
    classify_preccision=[]
    classify_recall=[]
    
    for track_length in range(5,13):
        tracks = torch.from_numpy(total_tracks[str(track_length)]).float()
        track_num, _, _ = tracks.shape
        pred_tracks = torch.zeros((track_num, track_length, 2))
        pred_tracks[:, :4, :] = tracks[:, :4, :-1]
        cls_label = tracks[:, -(track_length-4):, -1]
        cls_pred = torch.zeros((track_num,track_length-4))
        
        #inference
        model.eval()
        tmp_regression_rmse = []   
        tmp_regression_mae = []        
        for j in range(track_num):
            if j == 3:
                continue
            if j % 2 == 0 :
                origin_true_track = tracks[j:j+1,:,:].clone()
                # normalization
                origin_true_track[:,:,:2] = data_normal(origin_true_track[:,:,:2],physical_width)
                for k in range(track_length):
                    if k < 4:
                        continue
                    track_net = torch.zeros((1,5,5))
                    track_net[:,:,:2] = origin_true_track[:,k-4:k+1,:2]
                    track_net[:,1:,2:-1] = track_net[:,1:,:2] - track_net[:,:-1,:2]
                    track_net[:,:,-1:] = origin_true_track[:,k-4:k+1,-1:]
                    input_seq = track_net[:, :4 , :-1]
                    candidates = torch.reshape(track_net[:,-2: , :-1], (1, 8)) 
                    pred_regression, classification = model(input_seq, candidates)
                    classification = torch.softmax(classification,dim=1)
                    belong = torch.argmax(classification,dim=1)
                    cls_pred[j,:] = belong
                    if belong == 1 and (classification[0,1]-classification[0,0]).item() > 0.7:
                        pred_regression = inverse_data_normal(pred_regression[:,:2],physical_width)
                        true_regression = inverse_data_normal(track_net[-1,-1:,:2],physical_width)
                        tmp_regression_rmse.append(torch.sqrt(F.mse_loss(pred_regression,true_regression)).item())
                        tmp_regression_mae.append(torch.abs(pred_regression-true_regression).sum().item())

            else:
                origin_true_track = tracks[j-1:j,:,:].clone()
                origin_false_track = tracks[j:j+1,:,:].clone()
                origin_true_track[:,:,:2] = data_normal(origin_true_track[:,:,:2],physical_width)
                origin_false_track[:,:,:2] = data_normal(origin_false_track[:,:,:2],physical_width)
                for k in range(track_length):
                    if k < 4:
                        continue
                    track_net = torch.zeros((1,5,5))
                    track_net[:,:,:2] = torch.cat((origin_true_track[:,k-4:k,:2],origin_false_track[:,k:k+1,:2]),dim=1)
                    track_net[:,1:,2:-1] = track_net[:,1:,:2] - track_net[:,:-1,:2]
                    track_net[:,:,-1:] = torch.cat((origin_true_track[:,k-4:k,-1:],origin_false_track[:,k:k+1,-1:]),dim=1)
            
                    input_seq = track_net[:, :4 , :-1]
                    candidates = torch.reshape(track_net[:,-2: , :-1], (1, 8)) 
                    pred_regression, classification = model(input_seq, candidates)
                    classification = torch.softmax(classification,dim=1)
                    belong = torch.argmax(classification,dim=1)
                    cls_pred[j,:] = belong

                    if belong == 1 and (classification[0,1]-classification[0,0]).item() > 0.7:
                        pred_regression = inverse_data_normal(pred_regression[:,:2],physical_width)
                        true_regression = inverse_data_normal(track_net[-1,-1:,:2],physical_width)
                        tmp_regression_rmse.append(torch.sqrt(F.mse_loss(pred_regression,true_regression)).item())
                        tmp_regression_mae.append(torch.abs(pred_regression-true_regression).sum().item())
                        
        # regression index
        tmp_regression_rmse = torch.tensor(tmp_regression_rmse)
        tmp_regression_mae = torch.tensor(tmp_regression_mae)
        regression_rmse.append(torch.mean(tmp_regression_rmse).item())
        regression_mae.append(torch.mean(tmp_regression_mae).item())
        
        # confusion matrix
        tp, fp, tn, fn,accuracy,precision,recall = 0, 0, 0, 0, 0, 0, 0
        for track_num in range(len(cls_label)):
            for item in  range(cls_label.shape[1]):    
                if cls_label[track_num,item] == 1 and cls_pred[track_num,item] == 1:
                    tp += 1
                if cls_label[track_num,item] == 0 and cls_pred[track_num,item] == 1:
                    fp += 1
                if cls_label[track_num,item] == 0 and cls_pred[track_num,item] == 0:
                    tn += 1
                if cls_label[track_num,item] == 1 and cls_pred[track_num,item] == 0:
                    fn += 1    
        accuracy = (tp+tn) / (tp+fp+tn+fn)
        if tp+fp != 0:
            precision = tp / (tp+fp)
        if tp+fn != 0:
            recall = tp / (tp + fn)
        
        classify_accuracy.append(accuracy)
        classify_preccision.append(precision)
        classify_recall.append(recall)
        
    return regression_rmse,regression_mae,classify_accuracy,classify_preccision,classify_recall


def main():        
    model = Net()
    test_path = "./src/simulate/data/nonlinear/scenario_2/test.npy"
    model_path = "./src/model/gru.pth" 
    model.load_state_dict(torch.load(model_path))
    total_tracks = np.load(test_path, allow_pickle=True).item()
    
    # 15mm
    physical_width = 15
    physical_height = 15

    # pixel size
    pixel = 0.0192

    # model inference  
    rmse_list,mae_list,accuracy_list,precision_list,recall_list = model_infer(model, total_tracks, physical_width)
    
    rmse_histogram(rmse_list,mae_list)
    classify_histogram(accuracy_list,precision_list,recall_list)

if __name__ == "__main__":
    print("*"*60)
    print('Infer_origin.py working path is '+os.getcwd())
    print("*"*60)
    main()
