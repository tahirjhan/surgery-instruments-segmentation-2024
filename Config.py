import os
import torch
import time
import ml_collections


save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 333
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True
n_channels = 3
n_labels = 1

epochs = 200
#epochs = 2000
img_size = 256
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 10

pretrain = False

task_name = 'InstrumentsSeg'

learning_rate = 1e-3
batch_size = 16

model_name = 'CFFANet_2'


train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Test_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'



test_session = "Test_session_05.27_12h31"