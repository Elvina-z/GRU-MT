import sys
sys.path.append('src/utils/')
import torch
import numpy as np
import matplotlib.pyplot as plt
import common_util
import datetime

device = torch.device("cuda:0")


# The track is split into track segments of length 5 
def preprocess(path,physical_width):
    #  dictionary with different lentgth (5~13)
    total_tracks = np.load(path, allow_pickle=True).item()
    true_tracklets = torch.zeros((1, 5, 5)).to(device)
    false_tracklets = torch.zeros((1, 5, 5)).to(device)
    for i in range(5, 13):
        tracks = torch.from_numpy(total_tracks[str(i)]).to(device).float()
        tracks[:,:,:-1] = common_util.data_normal(tracks[:,:,:-1],physical_width)
        track_num , track_length, _ = tracks.shape
        num = int(track_num/2)
        true_tracks =  tracks[0::2]
        false_tracks = tracks[1::2]
        true_tmp_tracklets=torch.zeros((num,5,5)).to(device)
        false_tmp_tracklets=torch.zeros((num,5,5)).to(device)

        for k in range(5,track_length+1):
            true_tmp_tracklets[:,:,:2] = true_tracks[:,k-5:k,:2]
            true_tmp_tracklets[:,0,2:-1] = 0
            true_tmp_tracklets[:,1:,2:-1] = true_tracks[:,k-4:k,:-1]-true_tracks[:,k-5:k-1,:-1]
            true_tmp_tracklets[:,:,-1:] = true_tracks[:,k-5:k,-1:]
            true_tracklets = torch.cat((true_tracklets,true_tmp_tracklets),axis=0)

            origin_false_tracklets = torch.cat((true_tracks[:,k-5:k-1,:],false_tracks[:,k-1:k,:]),axis=1)
            false_tmp_tracklets[:,:,:2] = origin_false_tracklets[:,:,:2]
            false_tmp_tracklets[:,0,2:-1] = 0
            false_tmp_tracklets[:,1:,2:-1] = origin_false_tracklets[:,1:,:-1]-origin_false_tracklets[:,:-1,:-1]
            false_tmp_tracklets[:,:,-1:] = origin_false_tracklets[:,:,-1:]
            false_tracklets = torch.cat((false_tracklets,false_tmp_tracklets),axis=0)
    
    tracklets = torch.cat((true_tracklets[1:,:],false_tracklets[1:,:]),axis=0)
    
    input_seq = tracklets[:, :-1, :-1]
    candidates = tracklets[:, -2:, :-1]
    reg_label = torch.cat((true_tracklets[1:,-1,:],true_tracklets[1:,-1,:]),axis=0)[:,:-1]
    cls_label = tracklets[:, -1, -1].long()


    return input_seq,candidates,reg_label,cls_label


def adjust_learning_rate( optimizer, epoch, start_lr):
    lr = start_lr * (0.5 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def visual_loss( train_loss,val_loss,epoch, bs,type,flag):
    dateTime_p = datetime.datetime.now().date()
    str_p = datetime.datetime.strftime(dateTime_p,'%Y-%m-%d')
    epochs = range(1, len(train_loss) + 1)
    if type==1:
        plt.plot(
        epochs, train_loss, label="Train loss", color="red", linestyle="--",
    )
        plt.plot(
            epochs, val_loss, label="Validate loss", color="green", linestyle="--",
        )
        plt.title("Train and Validate loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            "./src/result/train/loss/"+flag+"_"+str_p+"_epoch_" + str(epoch + 1) + "_bs_" + str(bs) + "_all_.png", format="png"
        )
        plt.close()
    elif type==2:
        plt.plot(
        epochs, train_loss, label="Train reg loss", color="red", linestyle="--",
    )
        plt.plot(
            epochs, val_loss, label="Validate reg loss", color="green", linestyle="--",
        )
        plt.title("Regression loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            "./src/result/train/loss/"+flag+"_"+str_p+"_epoch_" + str(epoch + 1) + "_bs_" + str(bs)+"_reg_" + ".png", format="png"
        )
        plt.close()
    else:
        plt.plot(
        epochs, train_loss, label="Train cls loss", color="red", linestyle="--",
    )
        plt.plot(
            epochs, val_loss, label="Validate cls loss", color="green", linestyle="--",
        )
        plt.title("Classify loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            "./src/result/train/loss/"+flag+"_"+str_p+"_epoch_" + str(epoch + 1)  + "_bs_" + str(bs) +"_cls_"+".png", format="png"
        )
        plt.close()


def visual_regression(rmse,mae,bs,flag):
    dateTime_p = datetime.datetime.now().date()
    str_p = datetime.datetime.strftime(dateTime_p,'%Y-%m-%d')
    epochs = range(1, len(rmse) + 1)
    plt.plot(epochs, rmse, 'b', label='RMSE')
    plt.plot(epochs, mae, 'g', label='MAE')
    plt.title('Regression error')
    plt.xlabel("Epoch")
    plt.ylabel("mm")
    plt.legend()
    plt.savefig(
            "./src/result/train/error/"+flag+"_"+str_p+"_regression_epoch_" + str(len(rmse))  +"_reg_"+ "_bs_" + str(bs) + ".png", format="png"
        )
    # plt.show()
    plt.close()  


def visual_classification(acc,prec,rec,bs,flag):
    dateTime_p = datetime.datetime.now().date()
    str_p = datetime.datetime.strftime(dateTime_p,'%Y-%m-%d')
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, label='Accuraccy')
    plt.plot(epochs, prec, 'b', label='Precision')
    plt.plot(epochs, rec, 'g', label='Recall')
    plt.title('Classify result')
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(
            "./src/result/train/error/"+flag+"_"+str_p+"_classify_epoch_" + str(len(acc)) +"_cls_" + "_bs_" + str(bs) + ".png", format="png"
        )
    # plt.show()  
    plt.close()
        
