
python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --data_path='/ssd1/zhangfeihu/data/kitti/2015//testing/' \
                  --test_list='lists/kitti2015_test.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/finetune2_kitti2015_epoch_8.pth'
exit

python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --data_path='/media/feihu/Storage/stereo/kitti/testing/' \
                  --test_list='lists/kitti2012_test.list' \
                  --save_path='./result/' \
                  --kitti=1 \
                  --resume='./checkpoint/finetune2_kitti_epoch_8.pth'



