import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
import cv2 as cv

sys.path.append('src/utils/')


def zero_res(x):
    return x if x >= 0 else 0

def max_res(x, edge):
    return x if x <= edge else edge

# regression prediction  hsitogram 
def rmse_histogram(rmse_list,mae_list):
    dateTime_p = datetime.datetime.now().date()
    str_p = datetime.datetime.strftime(dateTime_p,'%Y-%m-%d')
    # plt.rcParams['font.sans-serif'] = ['SimHei']  
    x_list = np.arange(5,13)
    bar_width = 0.2  
    plt.xlabel('Track length')
    plt.ylabel('Classification value')
    plt.bar(x_list,rmse_list, bar_width, align='center', color='#d05451', label='RMSE')
    plt.bar(x_list + bar_width, mae_list, bar_width, align='center', color='#8da0cb', label='MAE')
    plt.xlabel('Track length')
    plt.ylabel('RMSE(mm)')
    plt.title('RMSE values corresponding to different track lengths')
    plt.legend()
    fig = plt.gcf()
    # plt.show()
    fig.savefig('./src/result/infer/regression_'+str_p+'_normal.png')
    plt.close()

# classification prediction histogram
def classify_histogram(accuracy_list,precision_list,recall_list):
    dateTime_p = datetime.datetime.now().date()
    str_p = datetime.datetime.strftime(dateTime_p,'%Y-%m-%d')
    # plt.rcParams['font.sans-serif'] = ['SimHei'] 
    x_list = np.arange(5,13)
    bar_width = 0.2  
    plt.xlabel('Track length')
    plt.ylabel('Classification value')
    plt.bar(x_list-bar_width,accuracy_list, bar_width, align='center', color='#66c2a5', label='accuracy')
    plt.bar(x_list, precision_list, bar_width, align='center', color='#8da0cb', label='precision')
    plt.bar(x_list+bar_width, recall_list, bar_width, align='center', color='#a31515', label='recall')
    plt.legend()
    plt.title('Classification index corresponding to different track length')
    fig = plt.gcf()
    # plt.show()
    fig.savefig('./src/result/infer/classify_'+str_p+'_normal.png')
    plt.close()


