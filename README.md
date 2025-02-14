


# TAPOR

TAPOR: 3D Hand Pose Reconstruction for IoT Interaction via Low-Cost Thermal Sensing

![Alt Text](figures/short_demo1(1).gif)

(Full Demo: https://www.youtube.com/watch?v=XCKol-EjH7Y)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/XCKol-EjH7Y/0.jpg)](https://www.youtube.com/watch?v=XCKol-EjH7Y)

## Environment Setup
```
conda tapor create -f environment.yml
```

## Dataset

To obtain the dataset and the pre-trained model weights for this project, please access the provided [link](https://drive.google.com/drive/folders/1qCkaUHxPGxaJgvPovI4fizwN5m1PztR9?usp=sharing) and save it in the root directory. 

## Running the Code

### Training Tapor 
```
python traintest.py -m tapor -e 200 -b 48 -fs 1 -ms 0 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 -mt 1 -ls jb -tqdm 1 

```
### Training other baseline models
```
python traintest.py -m mediapipe -e 200 -b 64 -fs 1 -ms 256 -t 0 -lr 0.001 -hm 0 -fo 0 -s 2 

python traintest.py -m baseline3d  -e 400 -b 64 -fs 1 -ms 200 -t 0 -lr 0.00001 -hm 1 -fo 1 -s 3 

python traintest.py -m mano -e 200 -b 64 -fs 1 -ms 200 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 
```

### Testing Tapor
```
nohup python inference.py -m tapor -wp tapor.pth -fs 1 -ms 0 -hm 0 -fo 1 -mt 1 -v 0 

```
### Testing other baseline models
```
python inference.py -m tapor -wp tapor.pth -fs 1 -ms 0 -hm 0 -fo 1 -mt 1 -v 0 

python inference.py -m mediapipe -wp mediapipe.pth -fs 1 -ms 256 -hm 0 -fo 1 -v 0 

python inference.py -m baseline3d -wp baseline3d.pth -fs 1 -ms 200 -hm 0 -fo 1 -v 0 

```

### Try NanoTapor on the IoT device
Please use the Firmware folder to flash the NanoTapor firmware on the ESP32S# device.

## Citation

Coming soon.
