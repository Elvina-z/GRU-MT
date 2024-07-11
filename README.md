# GRU-MT
Efficient Microbubble Trajectory Tracking in Ultrasound Localization Microscopy Using a Gated Recurrent Unit-Based Multitasking Temporal Neural Network

## Simulation datasets generation
```python
cd ./src/simulate/
python nonlinearMotionModel.py
```

## Training
```python
cd ./src/
python train.py
```

## Inference
```python
cd ./src/
python infer.py
```

## Citation
If you find this project useful for your research, please use the following BibTeX entry.

```
@ARTICLE{Zhang2024,
  author={Zhang, Yuting and Zhou, Wenjun and Huang, Lijie and Shao, Yongjie and Luo, Anguo and Luo, Jianwen and Peng, Bo},
  journal={IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control}, 
  title={Efficient Microbubble Trajectory Tracking in Ultrasound Localization Microscopy Using a Gated Recurrent Unit-Based Multitasking Temporal Neural Network}, 
  year={2024},
  volume={},
  number={},
  pages={},
  note={early access, doi: {10.1109/TUFFC.2024.3424955}},
  keywords={Ultrasonic imaging;Trajectory tracking;Imaging;Location awareness;Acoustics;Task analysis;Frequency control;Gated recurrent unit;Microbubble tracking;Nonlinear motion;Ultrasound localization microscopy;Ultrasound super-resolution imaging}
}
```
