from .dataset import DatasetFromList

def get_training_set(data_path, train_list, crop_size=[256,256], left_right=False, kitti=False, kitti2015=False, shift=0):
    return DatasetFromList(data_path, train_list,
                             crop_size, True, left_right, kitti, kitti2015, shift)


def get_test_set(data_path, test_list, crop_size=[256,256], left_right=False, kitti=False, kitti2015=False):
    return DatasetFromList(data_path, test_list,
                             crop_size, False, left_right, kitti, kitti2015)
