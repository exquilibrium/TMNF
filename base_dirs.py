import os

BASE_DIR_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_WEIGHTS_FOLDER = BASE_DIR_FOLDER+'/mmdetection/weights'

if BASE_DIR_FOLDER == '/home/chen/TMNF': # Home PC
    BASE_DATA_FOLDER = '/media/chen/76AECF8EAECF4579/data'
    BASE_VOC_FOLDER = '/media/chen/76AECF8EAECF4579/data'
elif BASE_DIR_FOLDER == '/home/chen_le/TMNF': # DLR PC
    BASE_DATA_FOLDER = '/volume/hot_storage/slurm_data/chen_le/ARCHES'
    BASE_VOC_FOLDER = '/volume/hot_storage/slurm_data/chen_le'
else:
    print('\n\n\nAdd your dataset paths!!!\n\n\n')

BASE_RESULTS_FOLDER = BASE_DIR_FOLDER + '/results'
BASE_PRETRAINED_FOLDER = BASE_DIR_FOLDER + '/pretrained'
