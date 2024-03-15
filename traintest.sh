
python traintest.py -m mediapipe -e 200 -b 64 -fs 1 -ms 256 -t 0 -lr 0.001 -hm 0 -fo 0 -s 2 

python traintest.py -m baseline3d  -e 400 -b 64 -fs 1 -ms 200 -t 0 -lr 0.00001 -hm 1 -fo 1 -s 3 

python traintest.py -m mano -e 200 -b 64 -fs 1 -ms 200 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 

python traintest.py -m tapor -e 200 -b 48 -fs 1 -ms 0 -t 0 -lr 0.0001 -hm 0 -fo 1 -s 0 -mt 1 -ls jb -tqdm 1 
