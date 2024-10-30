
python inference.py -m tapor -wp tapor.pth -fs 1 -ms 0 -hm 0 -fo 1 -mt 1 -v 0 

python inference.py -m mediapipe -wp mediapipe.pth -fs 1 -ms 256 -hm 0 -fo 1 -v 0 

python inference.py -m baseline3d -wp baseline3d.pth -fs 1 -ms 200 -hm 0 -fo 1 -v 0 

nohup python inference.py -m tapor -wp tapor.pth -fs 1 -ms 0 -hm 0 -fo 1 -mt 1 -v 0 

python nano_tapor_inference.py -wp nano_tapor.pth -fs 1 -fd 336 -ld 48 -bs 128 -dd 0 -v 1