CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
                --crop_height=240 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/ssd1/zhangfeihu/data/stereo/' \
                --training_list='lists/sceneflow_train.list' \
                --save_path='./checkpoint/sceneflow' \
                --resume='' \
                --model='GANet_deep' \
                --nEpochs=11 2>&1 |tee logs/log_train_sceneflow.txt

exit
#Fine tuning for kitti 2015
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
                --crop_height=240 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/data_scene_flow/training/' \
                --training_list='lists/kitti2015_train.list' \
                --save_path='./checkpoint/finetune_kitti2015' \
                --kitti2015=1 \
                --shift=3 \
                --resume='./checkpoint/sceneflow_epoch_10.pth' \
                --nEpochs=800 2>&1 |tee logs/log_finetune_kitti2015.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
                --crop_height=240 \
                --crop_width=1248 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/data_scene_flow/training/' \
                --training_list='lists/kitti2015_train.list' \
                --save_path='./checkpoint/finetune2_kitti2015' \
                --kitti2015=1 \
                --shift=3 \
                --lr=0.0001 \
                --resume='./checkpoint/finetune_kitti2015_epoch_800.pth' \
                --nEpochs=8 2>&1 |tee logs/log_finetune_kitti2015.txt

#Fine tuning for kitti 2012

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
                --crop_height=240 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/kitti/training/' \
                --training_list='lists/kitti2012_train.list' \
                --save_path='./checkpoint/finetune_kitti' \
                --kitti=1 \
                --shift=3 \
                --resume='./checkpoint/sceneflow_epoch_10.pth' \
                --nEpochs=800 2>&1 |tee logs/log_finetune2_kitti.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
                --crop_height=240 \
                --crop_width=1248 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/kitti/training/' \
                --training_list='lists/kitti2012_train.list' \
                --save_path='./checkpoint/finetune2_kitti' \
                --kitti=1 \
                --shift=3 \
                --lr=0.0001 \
                --resume='./checkpoint/finetune_kitti_epoch_800.pth' \
                --nEpochs=8 2>&1 |tee logs/log_finetune2_kitti.txt




