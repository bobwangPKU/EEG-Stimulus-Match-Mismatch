# model setting
kernel_size = 3
att_out_dim = 256
dropout = 0.5
model_name = 'BrainNetworkCL'
## training setting
opt = 'Adam'
lr_dic={'Adam':2e-4, 'SGD':1e-4}
num_epochs = 300
early_stop_num = 20
batch_size = 32
negsample_num = 32
device = 'cuda:1'
use_multi_band = False

# feature setting
seg_len = 5 # second
fs = 64 # Hz
feature_name = 'wav2vec14pca_gpt9cw5pca_envelope'
feature_dim_dict = {'wav2vec14pca': 64, 'gpt9cw5pca': 4, 'envelope': 1, 'mel': 10}
embeds, feat_frames_id = [],[]
valid = 1 # validation fold index


