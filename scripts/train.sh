GPU_ID=1
cd ..

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -net ghost1_0 -dataset pic-day-cam1
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -net ghost1_0 -dataset pic-day-cam2
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -net ghost1_0 -dataset pic-day-cam3
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -net ghost1_0 -dataset pic-day-cam4
