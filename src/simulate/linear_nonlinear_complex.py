import numpy as np
import cv2 as cv
import argparse
import os
import copy
import sys
from datetime import datetime
from data_util.data_visual import paint as visual
from data_util.operate_util import divide as divide

path = os.path.dirname(os.path.dirname(__file__))+r"\simulate"
sys.path.append(path)


# scenario 1
def scenario_1(physical_width, physical_height, frame, trackID, seed):

    np.random.seed(seed)
    # make sure those point not be on the image egde
    x = np.random.uniform(physical_width/8, physical_width-(physical_width/8))
    z = np.random.uniform(physical_height/8, physical_height-(physical_height/8))
    # velocity
    v = np.random.uniform(5.0,10.0)
    # inital phase 
    pha = np.random.uniform(0,1) * np.pi
    # angular velocity
    w = np.random.uniform(-5, 5) * np.pi   
    # acceleration
    a = 0.1
    # 1 = true(this point is belong to the track)   0 = false
    label = 1

    state_space = np.array([x, z, v, pha, w, a, label, frame, trackID]).reshape((1, 9))
    return state_space



# scenario 2
def scenario_2(physical_width, physical_height, frame, trackID, seed):

    np.random.seed(seed)
    x = np.random.uniform(physical_width/8, physical_width - (physical_width/8))
    z = np.random.uniform(physical_height/8, physical_height-(physical_height/8))
    v = np.random.uniform(10.0,15.0)
    pha = np.random.uniform(0,1) * np.pi
    w = np.random.uniform(-8, 8) * np.pi   
    label = 1
    a = 0.2

    state_space = np.array([x, z, v, pha, w, a, label, frame, trackID]).reshape((1, 9))
    return state_space


# scenario 3
def scenario_3(physical_width, physical_height, frame, trackID, seed):
    
    np.random.seed(seed)
    x = np.random.uniform(physical_width/8, physical_width - (physical_width/8)-1)
    z = np.random.uniform(physical_height/8, physical_height-(physical_height/8)-1)
    v = np.random.uniform(15.0,20.0)
    pha = np.random.uniform(0,2) * np.pi
    w = np.random.uniform(-10, 10) * np.pi   
    label = 1
    a = 0.5

    state_space = np.array([x, z, v, pha, w, a, label, frame, trackID]).reshape((1, 9))
    return state_space



# non linear motion
def non_linear_iterate(state_space, delta_t, frame):
    x = state_space[:, 0]
    z = state_space[:, 1]
    v = state_space[:, 2]
    pha = state_space[:, 3]
    w = state_space[:, 4]
    a = state_space[:, 5]
    trackID = state_space[:, 8]

    pha_tmp = (pha + w * delta_t) 
    r_tmp = v / w
    x = x + r_tmp * (np.sin(pha) - np.sin(pha_tmp)) + np.random.normal(0,0.005,1)
    z = z + r_tmp * (np.cos(pha_tmp) - np.cos(pha)) + np.random.normal(0,0.005,1)
    pha = pha_tmp
    v = v + a * delta_t
    w = v / r_tmp

    label = 1

    state_space = np.array([x.item(), z.item(), v.item(), pha.item(), w.item(), a.item(), label, frame, trackID.item()]).reshape((1, 9))
    return state_space


# linear motion
def linear_iterate(state_space, delta_t, frame):

    x = state_space[:, 0]
    z = state_space[:, 1]
    v = state_space[:, 2]
    pha = state_space[:, 3]
    w = state_space[:, 4]
    a = state_space[:, 5]
    trackID = state_space[:, 8]

    x = x + v*delta_t
    z = z + v*delta_t
    label = 1

    state_space = np.array([x.item(), z.item(), v.item(), pha.item(), w.item(), a.item(), label, frame, trackID.item()]).reshape((1, 9))
    return state_space


# To create false point  
def false_candidate_linear(state_space, delta_t, frame):
    
    x = state_space[:, 0] 
    z = state_space[:, 1]
    v = state_space[:, 2]
    pha = state_space[:, 3]
    w = state_space[:, 4]
    a = state_space[:, 5]

    x = x + v  * delta_t + np.random.uniform(-1, 1) 
    z = z + v  * delta_t + np.random.uniform(-1, 1) 

    label = 0

    state_space[:, 0] = x
    state_space[:, 1] = z
    state_space[:, 6] = label
    state_space[:, 7] = frame

    return state_space


# To create false point  
def false_candidate_nonlinear(state_space, delta_t, frame):
    
    x = state_space[:, 0]
    z = state_space[:, 1]
    v = state_space[:, 2]
    pha = state_space[:, 3]
    w = state_space[:, 4]
    a = state_space[:, 5]
    trackID = state_space[:, 8]

    pha_tmp = (pha + w * delta_t) 
    r_tmp = v / w
    x = x + r_tmp * (np.sin(pha) - np.sin(pha_tmp)) + np.random.normal(0,0.005,1)
    z = z + r_tmp * (np.cos(pha_tmp) - np.cos(pha)) + np.random.normal(0,0.005,1)
    pha = pha_tmp
    v = v + a * delta_t
    w = v / r_tmp

    label = 0

    state_space = np.array([x.item(), z.item(), v.item(), pha.item(), w.item(), a.item(), label, frame, trackID.item()]).reshape((1, 9))
    return state_space


def save_tracks(base_path, filenames, data_list):
    for filename, data in zip(filenames, data_list):
        path = f"{base_path}/{filename}"
        np.savetxt(path, data, delimiter=',')


# generate a track with length 
def generate_tracks(physical_width, physical_height, delta_t, nums, length, pixel, scenario, total_frame, trackID, max_disp_frame):

    img_widths = int(physical_width/pixel)
    img_height = int(physical_height/pixel)
    img = np.zeros((img_widths, img_height, 3), np.uint8)

    # Even numbers are nonlinear trajectories, odd numbers are linear trajectories
    tracks = np.zeros((nums*2, length, 9))  # Complete set of trajectories: [x, z, v, pha, w, a, label, frame, trackID]
    tracks_temp = np.zeros((nums*2, length, 9))  # Set of trajectories containing disappearing points
    for num in range(nums):
        trackID = trackID + 1
        seed = trackID + datetime.now().microsecond
        frame = np.random.randint(0, total_frame-length)  # start frame
        flag = False   # Indicates whether the trajectory has a disappearing point
        cnt = np.random.randint(0, length-4)  # Random generation of disappearing frames
        if cnt > max_disp_frame:
            cnt = max_disp_frame 
        start_disp_frame = np.random.randint(frame+5, frame+length-cnt+1)  # Randomly generate the number of starting disappearing frame

        for k in range(length):
            frame = frame + 1 
            # Make sure the first four frames of the trajectory are all correct
            if k < 4:
                if k == 0:
                    if scenario == '1':
                        ss = scenario_1(physical_width, physical_height, frame, trackID, seed)
                    if scenario == '2':
                        ss = scenario_2(physical_width, physical_height, frame, trackID, seed)
                    if scenario == '3':
                        ss = scenario_3(physical_width, physical_height, frame, trackID, seed)
                    track_true  = copy.copy(ss)
                    track_false = copy.copy(ss)
                else:
                    if trackID % 2 == 0:
                        ss = non_linear_iterate(ss, delta_t, frame)
                    else:
                        ss = linear_iterate(ss, delta_t, frame)
                    track_true = np.concatenate((track_true, ss), axis=0)
                    track_false = copy.copy(track_true)
                    track_true_temp = copy.copy(track_true) 
                    track_false_temp = copy.copy(track_false)
                
                true_width = int(ss[0, 0]/pixel)
                true_height = int(ss[0, 1]/pixel)
                img = visual(img, img_widths, img_height,true_width, true_height, "white")

                continue

            # Starting at frame five
            # Trajectories with IDs in multiples of 3 start disappearing from any frame
            if trackID % 3 == 0 and start_disp_frame <= k < start_disp_frame + cnt:
                flag = True  
                false_ss_temp = np.hstack((np.full((1, 7), -1), np.reshape(np.array([frame, trackID]),(1,2))))
                track_false_temp = np.concatenate((track_false_temp, false_ss_temp), axis=0) 
                ss_temp = np.hstack((np.full((1, 7), -1), np.reshape(np.array([frame, trackID]),(1,2))))
                track_true_temp = np.concatenate((track_true_temp, ss_temp), axis=0)

                if trackID % 2 == 0:
                    false_ss = false_candidate_linear(copy.copy(ss), delta_t, frame)
                else:
                    false_ss = false_candidate_nonlinear(ss, delta_t, frame)  
                track_false = np.concatenate((track_false, false_ss), axis=0)               
                if trackID % 2 == 0:
                    ss = non_linear_iterate(ss, delta_t, frame)
                else:
                    ss = linear_iterate(ss, delta_t, frame)
                track_true = np.concatenate((track_true, ss), axis=0)
                continue

            else:

                if trackID % 2 == 0:
                    ss = non_linear_iterate(ss, delta_t, frame)
                else:
                    ss = linear_iterate(ss, delta_t, frame)
                track_true = np.concatenate((track_true, ss), axis=0)
                track_true_temp = np.concatenate((track_true_temp, ss), axis=0)

                if trackID % 2 == 0:
                    false_ss = false_candidate_linear(copy.copy(ss), delta_t, frame)
                else:
                    false_ss = false_candidate_nonlinear(ss, delta_t, frame)
                track_false = np.concatenate((track_false, false_ss), axis=0) 
                track_false_temp = np.concatenate((track_false_temp, false_ss), axis=0) 


            # visual track motion
            false_width = int(false_ss[0, 0]/pixel)
            false_height = int(false_ss[0, 1]/pixel)
            true_width = int(ss[0, 0]/pixel)
            true_height = int(ss[0, 1]/pixel)

            if flag:
                img = visual(img, img_widths, img_height, false_width, false_height, "yellow")
                img = visual(img, img_widths, img_height, true_width, true_height, "blue")
                # cv.imshow("Track"+str(length), img)
                # cv.waitKey(-1)
            else:
                img = visual(img, img_widths, img_height, false_width, false_height, "red")
                img = visual(img, img_widths, img_height, true_width, true_height, "green")
            

        tracks[num*2, :, :] = track_true
        tracks[num*2+1, :, :] = track_false
        tracks_temp[num*2, :, :] =  track_true_temp
        tracks_temp[num*2+1, :, :] = track_false_temp

    cv.imshow("Track"+str(length), img)
    cv.waitKey(-1)

    return tracks, tracks_temp


def main(scenario):         
    
    # 15mm
    physical_width = 15
    physical_height = 15
    
    # tracks for train
    train_num = 500
    # tracks for validate
    validate_num = 300
    # tracks for test
    test_num = 200

    track_nums = train_num + validate_num + test_num

    if scenario == '1':
        test_path = './src/simulate/data/complex/scenario_1/'
    elif scenario == '2':       
        test_path = './src/simulate/data/complex/scenario_2'        
    elif scenario == '3':       
        test_path = './src/simulate/data/complex/scenario_3'
    else:
        print("scenario error")
        exit(0)
    
    delta_t = 0.02
    pixel = 0.0192

    total_frame = 50
    trackID = 0

    total_true_tracks = np.zeros((1, 9))
    total_false_tracks = np.zeros((1, 9))
    total_true_tracks_temp = np.zeros((1, 9))
    total_false_tracks_temp = np.zeros((1, 9))

    max_disp_frame = 3

    for track_length in range(5, 13):       
        # Generate trajectories
        tracks, tracks_temp = generate_tracks(physical_width, physical_height, delta_t, track_nums, track_length, pixel, scenario, total_frame, trackID, max_disp_frame)  
        # Segmentation of trajectories for train
        # train_tracks[str(track_length)], val_tracks[str(track_length)], test_tracks[str(track_length)] = divide(tracks, train_num, validate_num, test_num)  
        trackID = trackID + track_nums

        # Full trajectories
        true_tracks = np.vstack(tracks[::2])  # true trajectories                        
        false_tracks = np.vstack(tracks[1::2])  # false trajectories 
        total_true_tracks = np.vstack((total_true_tracks, true_tracks))  
        total_false_tracks = np.vstack((total_false_tracks, false_tracks))  
        # Trajectories with disappearing points
        true_tracks_temp = np.vstack(tracks_temp[::2])  # true trajectories                    
        false_tracks_temp = np.vstack(tracks_temp[1::2])  # false trajectories 
        total_true_tracks_temp = np.vstack((total_true_tracks_temp, true_tracks_temp))  
        total_false_tracks_temp = np.vstack((total_false_tracks_temp, false_tracks_temp))


    data_list = [
    total_true_tracks[1:],
    total_false_tracks[1:],
    total_true_tracks_temp[1:],
    total_false_tracks_temp[1:]
    ]

    filenames = [
    "total_true_tracks.txt",
    "total_true_tracks_temp.txt",
    "total_false_tracks.txt",
    "total_false_tracks_temp.txt"
    ]

    save_tracks(test_path, filenames, data_list)
    


if __name__ == '__main__':
    print("*"*60)
    print('Working path is '+os.getcwd())
    print("*"*60)
    parser = argparse.ArgumentParser(prog='input',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="ZhangYT")
                                        
    # Scenario 2 is used by default
    parser.add_argument('-s', type=str, help='Choose scenario.', default="2")
    args = parser.parse_args()
    main(args.s)
