CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/ssd1/zhangfeihu/data/kitti2015/training/' \
                  --test_list='lists/kitti2015_train.list' \
                  --save_path='./result/' \
                  --resume='./checkpoint/kitti2015_final.pth' \
                  --threshold=3.0 \
                  --kitti2015=1
# 2>&1 |tee logs/log_evaluation.txt
exit
CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/ssd1/zhangfeihu/data/kitti2012/training/' \
                  --test_list='lists/kitti2012_train.list' \
                  --save_path='./result/' \
                  --resume='./checkpoint/kitti_final.pth' \
                  --threshold=3.0 \
                  --kitti=1
# 2>&1 |tee logs/log_evaluation.txt
exit
CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=576 \
                  --crop_width=960 \
                  --max_disp=192 \
                  --data_path='/ssd1/zhangfeihu/data/sceneflow/' \
                  --test_list='lists/sceneflow_test.list' \
                  --save_path='./result/' \
                  --resume='./checkpoint/sceneflow_epoch_10.pth' \
                  --threshold=1.0 
# 2>&1 |tee logs/log_evaluation.txt
exit

