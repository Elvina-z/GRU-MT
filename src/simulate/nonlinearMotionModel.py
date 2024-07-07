import numpy as np
import cv2 as cv
import argparse
import os
import copy
import sys

from data_util.data_visual import paint as visual
from data_util.operate_util import divide as divide

path = os.path.dirname(os.path.dirname(__file__))+r"\simulate"
sys.path.append(path)

# scenario 1
def scenario_1(physical_width, physical_height):

    # make sure those point not be on the image egde =>  physical_width/8, physical_width - (physical_width/8)
    x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8), 1))
    z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8), 1))
    # velocity
    v = np.random.uniform(5.0, 10.0)
    # inital phase 
    pha = np.random.uniform(0,1) * np.pi
    # angular velocity 
    w = np.random.uniform(-5.0, 5.0) * np.pi
    # acceleration
    a = 0.1
    # 1 = true(this point is belong to the track)   0 = false
    label = 1

    state_space = np.array([x, z, v, pha, w, a, label]).reshape((1, 7))
    return state_space


# scenario 2
def scenario_2(physical_width, physical_height):

    # make sure those point not be on the image egde =>  physical_width/8, physical_width - (physical_width/8)
    x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8), 1))
    z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8), 1))
    # velocity
    v = np.random.uniform(10.0,15.0)
    # inital phase
    pha = np.random.uniform(0,1) * np.pi
    # angular velocity 
    w = np.random.uniform(-8, 8) * np.pi   
    # acceleration
    a = 0.2
    # 1 = true(this point is belong to the track)   0 = false
    label = 1

    state_space = np.array([x, z, v, pha, w, a, label]).reshape((1, 7))
    return state_space


def scenario_3(physical_width, physical_height):

    # make sure those point not be on the image egde =>  physical_width/8, physical_width - (physical_width/8)
    x = int(np.random.randint(physical_width/8, physical_width - (physical_width/8), 1))
    z = int(np.random.randint(physical_height/8, physical_height-(physical_height/8), 1))
    # velocity
    v = np.random.uniform(15.0,20.0)
    # inital phase 
    pha = np.random.uniform(0,2) * np.pi
    # angular velocity -10Π/s~10Π/s.     w*delta_t  = w*0.02 
    w = np.random.uniform(-10, 10) * np.pi       
    # acceleration
    a = 0.5 
    # 1 = true(this point is belong to the track)   0 = false
    label = 1

    state_space = np.array([x, z, v, pha, w, a, label]).reshape((1, 7))
    return state_space



# non linear motion
def non_linear_iterate(state_space, delta_t):
    x = state_space[:, 0]
    z = state_space[:, 1]
    v = state_space[:, 2]
    pha = state_space[:, 3]
    w = state_space[:, 4]
    a = state_space[:, 5]

    pha_tmp = (pha + w * delta_t) 
    r_tmp = v / w
    x = x + r_tmp * (np.sin(pha) - np.sin(pha_tmp)) + np.random.normal(0,0.005,1)
    z = z + r_tmp * (np.cos(pha_tmp) - np.cos(pha)) + np.random.normal(0,0.005,1)
    pha = pha_tmp
    v = v + a * delta_t
    w = v / r_tmp
    label = 1

    state_space = np.array([x.item(), z.item(), v.item(), pha.item(), w.item(), a.item(), label]).reshape((1, 7))
    return state_space


# linear motion
def linear_iterate(state_space, delta_t):

    x = state_space[:, 0]
    z = state_space[:, 1]
    v = state_space[:, 2]
    pha = state_space[:, 3]
    w = state_space[:, 4]
    a = state_space[:, 5]
 
    x = x + v*delta_t
    z = z + v*delta_t
    label = 1

    state_space = np.array([x.item(), z.item(), v.item(), pha.item(), w.item(), a.item(), label]).reshape((1, 7))
    return state_space


# To create false point
def false_candidate(state_space):
    delta_t = 0.2
    
    x = state_space[:, 0] 
    z = state_space[:, 1]
    v = state_space[:, 2]
    label = state_space[:, -1]

    # Sudden change of velocity to simulate false point
    x = x + v * delta_t
    z = z + v * delta_t
    label = 0

    x = x + v  * delta_t + np.random.uniform(-1, 1) 
    z = z + v  * delta_t + np.random.uniform(-1, 1) 

    state_space[:, 0] = x
    state_space[:, 1] = z
    state_space[:, -1] = label

    return state_space




# generate a track with length 
def generate_tracks(physical_width, physical_height, delta_t, nums, length, pixel, scenario):


    tracks = np.zeros((nums*2, length, 7))
    for num in range(nums):
        for k in range(length):
            # keep the first four points of the track are always correct
            if k < 4:
                if k == 0:
                    if scenario == '1':
                        ss = scenario_1(physical_width, physical_height)
                    if scenario == '2':
                        ss = scenario_2(physical_width, physical_height)
                    if scenario == '3':
                        ss = scenario_3(physical_width, physical_height)
                    track_true  = copy.copy(ss)
                    track_false = copy.copy(ss)
                else:
                    if num % 2 == 0:
                        ss = non_linear_iterate(ss, delta_t)
                    else:
                        ss = linear_iterate(ss, delta_t)
                    track_true = np.concatenate((track_true, ss), axis=0)
                    track_false = copy.copy(track_true)

                continue
            
            # use last correct point to createfalse point
            false_ss = false_candidate(copy.copy(ss))
            track_false = np.concatenate((track_false, false_ss), axis=0) 
            
            # create correct point
            if num % 2 == 0:
                ss = non_linear_iterate(ss, delta_t)
            else:
                ss = linear_iterate(ss, delta_t)

            track_true = np.concatenate((track_true, ss), axis=0)


        tracks[num*2, :, :] = track_true
        tracks[num*2+1, :, :] = track_false
    return tracks


def main(scenario):
    seed = 2
    np.random.seed(seed)            
    
    # 15mm
    physical_width = 15
    physical_height = 15
    
    # tracks for train
    train_num = 4000
    # tracks for validate
    validate_num = 1300
    # tracks for test
    test_num = 1300
    track_nums = train_num + validate_num + test_num
    if scenario == '1':
        train_path = './src/simulate/data/nonlinear/scenario_1/train'
        val_path = './src/simulate/data/nonlinear/scenario_1/val'
        test_path = './src/simulate/data/nonlinear/scenario_1/test'
    elif scenario == '2':       
        train_path = './src/simulate/data/nonlinear/scenario_2/train'
        val_path = './src/simulate/data/nonlinear/scenario_2/val'
        test_path = './src/simulate/data/nonlinear/scenario_2/test'
    elif scenario == '3':       
        train_path = './src/simulate/data/nonlinear/scenario_3/train'
        val_path = './src/simulate/data/nonlinear/scenario_3/val'
        test_path = './src/simulate/data/nonlinear/scenario_3/test'
    else:
        print("scenario error")
        exit(0)
    # 20ms
    delta_t = 0.02
    pixel = 0.0192

    train_tracks = {}
    val_tracks = {}
    test_tracks = {}

    for track_length in range(5, 13):
        tracks = generate_tracks(physical_width, physical_height, delta_t, track_nums, track_length, pixel ,scenario)
        train_tracks[str(track_length)], val_tracks[str(track_length)], test_tracks[str(track_length)] = divide(tracks, train_num, validate_num, test_num)
    
    np.save(train_path,train_tracks)
    np.save(val_path,val_tracks)
    np.save(test_path,test_tracks)


if __name__ == '__main__':
    print("*"*60)
    print('Working path is '+os.getcwd())
    print("*"*60)
    parser = argparse.ArgumentParser(prog='input',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="ZhangYT")
                                        
    # Scenario 2 is used by default
    parser.add_argument('-s', type=str, help='Choose scenario.', default="3")
    args = parser.parse_args()
    main(args.s)
