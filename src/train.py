import time
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import utils.train_util as train_util
import utils.common_util as common_util
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping

from net.net_gru import NetGRU as Net
model_name = "./src/model/gru.pth"

train_path = "./src/simulate/data/nonlinear/scenario_1/train.npy"
val_path = "./src/simulate/data/nonlinear/scenario_1/val.npy"

device = torch.device("cuda:0")

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, input_seq, candidates, reg_label, cls_label) -> None:
        self.input_seq = input_seq
        self.candidate = candidates
        self.reg_label = reg_label
        self.cls_label = cls_label

    def __len__(self):
        return self.input_seq.shape[0]

    def __getitem__(self, index):
        input_seq = self.input_seq[index]
        candidates = self.candidate[index]
        reg_label = self.reg_label[index]
        cls_label = self.cls_label[index]
        return input_seq, candidates, reg_label, cls_label

    
def joint_loss(reg_label, cls_label,pred_reg, pred_cls):
    reg_func = nn.SmoothL1Loss()
    cls_func = nn.CrossEntropyLoss()
    reg_loss = reg_func(pred_reg,reg_label)
    cls_loss = cls_func(pred_cls,cls_label)
    loss = torch.add(0.9*reg_loss,0.1*cls_loss)
    return loss,reg_loss,cls_loss


def train(train_loader, model, optimizer):

    model.train()
    total_loss,reg_loss,cls_loss = 0,0,0
    TP, FP, TN, FN =  0,0,0,0
    rmse_list = []
    mae_list = []

    accuracy = 0
    precision = 0
    recall = 0

    for input_seq, candidates, reg_label, cls_label in train_loader:
        candidates = torch.reshape(candidates, (candidates.shape[0], 8))
        pred_reg,pred_cls = model.forward(input_seq, candidates)

        # regression
        rmse_list.append(common_util.inverse_data_normal(F.mse_loss(reg_label,pred_reg).item(),15)*1000)
        mae_list.append(common_util.inverse_data_normal(torch.mean(torch.abs(reg_label-pred_reg)).item(),15)*1000)
        # classification
        TP, FP, TN, FN = common_util.confusion_matrix(cls_label,pred_cls,TP, FP, TN, FN)
        # loss function
        tmp_total_loss,tmp_reg_loss,tmp_cls_loss = joint_loss(reg_label, cls_label,pred_reg, pred_cls)
        
        total_loss += tmp_total_loss
        reg_loss += tmp_reg_loss
        cls_loss += tmp_cls_loss

        optimizer.zero_grad()
        tmp_total_loss = tmp_total_loss.requires_grad_()
        tmp_total_loss.backward()
        optimizer.step()

    total_loss = torch.mean(total_loss).item()
    reg_loss = torch.mean(reg_loss).item()
    cls_loss = torch.mean(cls_loss).item()

    rmse = torch.mean(torch.tensor(rmse_list))
    mae = torch.mean(torch.tensor(mae_list))

    accuracy = (TP+TN) / (TP+FP+TN+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP + FN)

    print("train")
    print("trian_loss={},train_reg_loss = {},train_cls_loss = {}".format(round(total_loss,2),round(reg_loss,2),round(cls_loss,2)))
    print("rmse = {},mae= {}".format(rmse,mae))
    print("TP={},FP={},TN={},FN={},accuracy={:.2%},precision={:.2%},recall = {:.2%}".format(TP,FP,TN,FN,accuracy,precision,recall))
    return total_loss,reg_loss,cls_loss,rmse,mae,accuracy,precision,recall


def validate(val_loader, model):
    
    model.eval()

    total_loss,reg_loss,cls_loss = 0,0,0
    rmse_list = []
    mae_list = []
    TP, FP, TN, FN =  0,0,0,0
    accuracy = 0
    precision = 0
    recall = 0

    with torch.no_grad():
        for input_seq, candidates, reg_label, cls_label in val_loader:
            candidates = torch.reshape(
                candidates, (candidates.shape[0], 8)
            )
            pred_reg,pred_cls = model.forward(input_seq, candidates)

            # regression
            rmse_list.append(common_util.inverse_data_normal(F.mse_loss(reg_label,pred_reg).item(),15)*1000)
            mae_list.append(common_util.inverse_data_normal(torch.mean(torch.abs(reg_label-pred_reg)).item(),15)*1000)
            # classification
            TP, FP, TN, FN = common_util.confusion_matrix(cls_label,pred_cls,TP, FP, TN, FN)

            # loss function
            tmp_total_loss,tmp_reg_loss,tmp_cls_loss = joint_loss(reg_label, cls_label,pred_reg,pred_cls)
            
            total_loss += tmp_total_loss
            reg_loss += tmp_reg_loss
            cls_loss += tmp_cls_loss

        total_loss = torch.mean(total_loss).item()
        reg_loss = torch.mean(reg_loss).item()
        cls_loss = torch.mean(cls_loss).item()

        rmse = torch.mean(torch.tensor(rmse_list))
        mae = torch.mean(torch.tensor(mae_list))

        accuracy = (TP+TN) / (TP+FP+TN+FN)
        precision = TP / (TP+FP)
        recall = TP / (TP + FN)

        print("validate")
        print("val_loss={},val_reg_loss={},val_cls_loss={}".format(round(total_loss,2),round(reg_loss,2),round(cls_loss,2)))
        print("rmse = {},mae= {}".format(rmse,mae))
        print("TP={},FP={},TN={},FN={},accuracy={:.2%},precision={:.2%},recall = {:.2%}".format(TP,FP,TN,FN,accuracy,precision,recall))
        
        return total_loss,reg_loss,cls_loss


def main():
    model = Net().to(device)

    epochs = 800
    start_lr = 1e-4
    batch_size = 1600
    optimizer = optim.Adam(model.parameters())
    width = 15
    flag = 'nonlinear'

    '''
    input_seq = left 4 point sequence
    candidates = right 2 points
    reg_label = ground truth regression label 
    cls_label = ground truth classification label 
    '''
    print("********Data preprocessing beginning : divide tracks into tracklets**********")
    input_seq, candidates, reg_label, cls_label =  train_util.preprocess(train_path,width)
    train_dataset = GetLoader(input_seq, candidates, reg_label, cls_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    val_seq, val_candidates, val_reg_label, val_cls_label =  train_util.preprocess(val_path,width)
    val_dataset = GetLoader(val_seq, val_candidates, val_reg_label, val_cls_label)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    print("********Data preprocessing finished**********")

    train_res = []
    train_reg_res = []
    train_cls_res = []

    val_res = []
    val_reg_res = []
    val_cls_res = []

    rmse_res = []
    mae_res = []
    
    acc_res = []
    precision_res = []
    recall_res = []


    early_stopping = EarlyStopping(patience=9, verbose=True)

    for epoch in range(epochs):
        start = time.time()
        train_util.adjust_learning_rate(optimizer, epoch, start_lr)
        print(
            "Epoch:{}  Lr:{:.2E}".format(
                epoch+1, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        # train
        train_loss, train_reg_loss,train_cls_loss,rmse,mae,accuracy,precision,recall = train(train_loader, model, optimizer)
        # validate
        val_loss,val_reg_loss,val_cls_loss = validate(val_loader, model)

        train_res.append(train_loss)
        train_reg_res.append(train_reg_loss)
        train_cls_res.append(train_cls_loss)

        val_res.append(val_loss)
        val_reg_res.append(val_reg_loss)
        val_cls_res.append(val_cls_loss)

        rmse_res.append(rmse)
        mae_res.append(mae)

        acc_res.append(accuracy)
        precision_res.append(precision)
        recall_res.append(recall)

        early_stopping(val_loss)
        

        if early_stopping.early_stop:
            print("Early stopping")
            break
        end = time.time() - start
        print(
            "one epoch time = {:.0f}m {:.0f}s".format(
                 end // 60 % 60, end % 60
            )
        )
        print('*'*60)
        print('*'*60)
    

    torch.save(model.state_dict(), model_name)
    
    # visual total_loss  regresssion_loss and  classification_loss  
    train_util.visual_loss(train_res, val_res, epoch, batch_size,1,flag)
    train_util.visual_loss(train_reg_res, val_reg_res, epoch, batch_size,2,flag)
    train_util.visual_loss(train_cls_res, val_cls_res, epoch, batch_size,3,flag)

    # visual regression error
    train_util.visual_regression(rmse_res,mae_res,batch_size,flag)

    # visual classification result
    train_util.visual_classification(acc_res,precision_res,recall_res,batch_size,flag)


if __name__ == "__main__":
    start = time.time()
    print('Trian_orgin.py working path is '+os.getcwd())
    main()
    end = time.time() - start
    print(
        "total time ={:.0f}h {:.0f}m {:.0f}s".format(
            end // 60 // 60, end // 60 % 60, end % 60
        )
    )
