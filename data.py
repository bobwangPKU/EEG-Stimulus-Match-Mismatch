import os
import config as cfg
from tqdm import tqdm
import os.path as op
import numpy as np
from torch.utils.data import Dataset
import glob
from mne.filter import filter_data



def get_stimulus_feat(feature_name='wav2vec14pca', seg_len=5, fs=64, std_feat='std'):
    # read all stimulus features
    all_stimulus_files = glob.glob(os.path.join(cfg.processed_stimuli_dir, "*"))
    all_stimulus_files = [x for x in all_stimulus_files if f'{feature_name}.npy' in x]
    # all_stimulus_files = [x for x in all_stimulus_files if 'test1' not in x]
    feat_dict = {}
    seg_len_emb = int(fs * seg_len)
    for stimulus_file in tqdm(all_stimulus_files):
        stimulus_name = os.path.basename(stimulus_file).split("_-_")[0]
        stimulus_feat = np.load(stimulus_file)
        stimulus_feat = stimulus_feat.astype(np.float32)
        frame_num = len(stimulus_feat) // seg_len_emb
        stimulus_feat = stimulus_feat[:frame_num * seg_len_emb]
        stimulus_feat_frames = stimulus_feat.reshape(frame_num, seg_len_emb, -1)
        EEG_sub = stimulus_feat_frames
        if std_feat=='std':
            EEG_sub_std = np.std(EEG_sub, axis=1)
            EEG_sub_mean = np.mean(EEG_sub, axis=1)
            EEG_sub = (EEG_sub - EEG_sub_mean[:, None, :]) / (EEG_sub_std[:, None, :] + 1e-8)
        elif std_feat == 'rbstd':
            center = np.median(EEG_sub, axis=1)
            q1 = np.percentile(EEG_sub, 25, axis=1)
            q3 = np.percentile(EEG_sub, 75, axis=1)
            scaler = q3 - q1
            scaler = scaler.astype(np.float32)
            EEG_sub = (EEG_sub - center[:, None, :]) / (scaler[:, None, :] + 1e-8)
            EEG_sub = np.clip(EEG_sub, -20, 20)
        else:
            ValueError('std_eeg should be std or rbstd')
        feat_dict[stimulus_name] = EEG_sub
    feat_keys = sorted(feat_dict.keys())
    feat_frames = np.concatenate([feat_dict[k] for k in feat_keys], axis=0)
    feat_frames_id = np.arange(feat_frames.shape[0])
    feat_frames_num = [feat_dict[k].shape[0] for k in feat_keys]
    feat_frames_id_split = np.split(feat_frames_id, np.cumsum(feat_frames_num)[0:-1])
    feat_frames_id_dict = {k: v for k, v in zip(feat_keys, feat_frames_id_split)}
    return feat_dict, feat_frames_id_dict, feat_keys, feat_frames


def get_EEG_emb_feat(sub_ls, feat_dict, feat_frames_id_dict, feat_keys, seg_len=5, fs=64, std_eeg='std', phase='train'):
    seg_len_emb = int(fs * seg_len)
    EEG_dict = {}
    EEG_feat_index_dict = {}
    Sub_id_dict = {}
    Stimulus_index_dict = {}
    all_subjects = [op.join(cfg.processed_eeg_dir, 'sub-{:03d}'.format(i)) for i in sub_ls]
    nb_subjects = len(all_subjects)
    train_stimulus_filename = []
    # Loop over subjects
    for subject_path in tqdm(all_subjects):
        subject = os.path.basename(subject_path)
        # Find all recordings
        all_recordings = glob.glob(os.path.join(subject_path, "*", "*.npy"))
        EEG_sub = []
        EEG_feat_index_sub = []
        stimulus_index_sub = []
        for recording in all_recordings:
            eeg = np.load(recording)
            eeg = eeg.astype(np.float32)
            eeg = np.swapaxes(eeg, 0, 1)
            eeg = eeg[:, :64]
            stimulus_filename = recording.split('_eeg.')[0].split('-audio-')[1]
            if stimulus_filename == 'audiobook_1' and phase == 'train':
                continue
            train_stimulus_filename.append(stimulus_filename)
            stimulus_feat = feat_dict[stimulus_filename]
            stimulus_feat_len = stimulus_feat.shape[0]*stimulus_feat.shape[1]
            if len(eeg) > stimulus_feat_len:
                eeg = eeg[:stimulus_feat_len]
            elif len(eeg) < stimulus_feat_len:
                eeg = np.concatenate([eeg, np.zeros((stimulus_feat_len - len(eeg), 64))], axis=0)
            eeg_frames = eeg.reshape(stimulus_feat.shape[0], seg_len_emb, -1)
            EEG_sub.append(eeg_frames)
            EEG_feat_index_sub.append(feat_frames_id_dict[stimulus_filename])
            stimulus_index_sub.append(feat_keys.index(stimulus_filename)*np.ones(len(eeg_frames), dtype=np.int64))
        EEG_sub = np.concatenate(EEG_sub, axis=0)
        if std_eeg=='std':
            EEG_sub_std = np.std(EEG_sub, axis=1)
            EEG_sub_mean = np.mean(EEG_sub, axis=1)
            EEG_sub = (EEG_sub - EEG_sub_mean[:, None, :]) / (EEG_sub_std[:, None, :] + 1e-8)
        elif std_eeg == 'rbstd':
            center = np.median(EEG_sub, axis=1)
            q1 = np.percentile(EEG_sub, 25, axis=1)
            q3 = np.percentile(EEG_sub, 75, axis=1)
            scaler = q3 - q1
            scaler = scaler.astype(np.float32)
            EEG_sub = (EEG_sub - center[:, None, :]) / (scaler[:, None, :] + 1e-8)
            EEG_sub = np.clip(EEG_sub, -20, 20)
        else:
            ValueError('std_eeg should be std or rbstd')
        EEG_dict[subject] = EEG_sub.astype(np.float32)
        EEG_feat_index_dict[subject] = np.concatenate(EEG_feat_index_sub, axis=0)
        Sub_id_dict[subject] = np.ones(len(EEG_feat_index_dict[subject]), dtype=np.int64) * (int(subject.split('-')[-1]) - 1)
        Stimulus_index_dict[subject] = np.concatenate(stimulus_index_sub, axis=0)

    sub_frames_id = [np.arange(len(EEG_dict[k])) for k in EEG_dict.keys()]
    sub_frames_subid = [v for k, v in Sub_id_dict.items()]
    sub_frames_id = np.concatenate(sub_frames_id, axis=0)
    sub_frames_subid = np.concatenate(sub_frames_subid, axis=0)
    id2dict = np.concatenate([sub_frames_subid[:, None], sub_frames_id[:, None]], axis=1)
    return EEG_dict, EEG_feat_index_dict, Sub_id_dict, Stimulus_index_dict, id2dict, train_stimulus_filename



def get_EEG_emb_feat_test(sub_ls, feat_dict, feat_frames_id_dict, feat_keys, train_stimulus_filename, seg_len=5, fs=64, std_eeg='std'):
    seg_len_emb = int(fs * seg_len)
    EEG_dict = {}
    EEG_feat_index_dict = {}
    Sub_id_dict = {}
    Stimulus_index_dict = {}
    all_subjects = [op.join(cfg.processed_eeg_dir, 'sub-{:03d}'.format(i)) for i in sub_ls]
    nb_subjects = len(all_subjects)
    # Loop over subjects
    for subject_path in tqdm(all_subjects):
        subject = os.path.basename(subject_path)
        # Find all recordings
        all_recordings = glob.glob(os.path.join(subject_path, "*", "*.npy"))
        EEG_sub = []
        EEG_feat_index_sub = []
        stimulus_index_sub = []
        for recording in all_recordings:
            eeg = np.load(recording)
            eeg = eeg.astype(np.float32)
            eeg = np.swapaxes(eeg, 0, 1)
            eeg = eeg[:, :64]
            stimulus_filename = recording.split('_eeg.')[0].split('-audio-')[1]
            if stimulus_filename in train_stimulus_filename:
                continue
            stimulus_feat = feat_dict[stimulus_filename]
            stimulus_feat_len = stimulus_feat.shape[0]*stimulus_feat.shape[1]
            if len(eeg) > stimulus_feat_len:
                eeg = eeg[:stimulus_feat_len]
            elif len(eeg) < stimulus_feat_len:
                eeg = np.concatenate([eeg, np.zeros((stimulus_feat_len - len(eeg), 64))], axis=0)
            eeg_frames = eeg.reshape(stimulus_feat.shape[0], seg_len_emb, -1)
            EEG_sub.append(eeg_frames)
            EEG_feat_index_sub.append(feat_frames_id_dict[stimulus_filename])
            stimulus_index_sub.append(feat_keys.index(stimulus_filename)*np.ones(len(eeg_frames), dtype=np.int64))
        if len(EEG_sub) == 0:
            continue
        EEG_sub = np.concatenate(EEG_sub, axis=0)
        if std_eeg=='std':
            EEG_sub_std = np.std(EEG_sub, axis=1)
            EEG_sub_mean = np.mean(EEG_sub, axis=1)
            EEG_sub = (EEG_sub - EEG_sub_mean[:, None, :]) / (EEG_sub_std[:, None, :] + 1e-8)
        elif std_eeg == 'rbstd':
            center = np.median(EEG_sub, axis=1)
            q1 = np.percentile(EEG_sub, 25, axis=1)
            q3 = np.percentile(EEG_sub, 75, axis=1)
            scaler = q3 - q1
            scaler = scaler.astype(np.float32)
            EEG_sub = (EEG_sub - center[:, None, :]) / (scaler[:, None, :] + 1e-8)
            EEG_sub = np.clip(EEG_sub, -20, 20)
        else:
            ValueError('std_eeg should be std or rbstd')
        EEG_dict[subject] = EEG_sub.astype(np.float32)
        EEG_feat_index_dict[subject] = np.concatenate(EEG_feat_index_sub, axis=0)
        Sub_id_dict[subject] = np.ones(len(EEG_feat_index_dict[subject]), dtype=np.int64) * (int(subject.split('-')[-1]) - 1)
        Stimulus_index_dict[subject] = np.concatenate(stimulus_index_sub, axis=0)
    # sub_frames_num = [EEG_dict[k].shape[0] for k in EEG_dict.keys()]
    # sub_frames_id_split = np.split(np.arange(np.sum(sub_frames_num)), np.cumsum(sub_frames_num)[0:-1])
    sub_frames_id = [np.arange(len(EEG_dict[k])) for k in EEG_dict.keys()]
    sub_frames_subid = [v for k, v in Sub_id_dict.items()]
    sub_frames_id = np.concatenate(sub_frames_id, axis=0)
    sub_frames_subid = np.concatenate(sub_frames_subid, axis=0)
    id2dict = np.concatenate([sub_frames_subid[:, None], sub_frames_id[:, None]], axis=1)
    return EEG_dict, EEG_feat_index_dict, Sub_id_dict, Stimulus_index_dict, id2dict



# Define dataset
class KUL_dataset(Dataset):
    def __init__(self, EEG_dict, EEG_feat_index_dict, Sub_id_dict,
                 Stimulus_index_dict,
                 id2dict, sub_names,
                 feat_keys, use_multi_band=False):

        self.band_limits = [0, 4, 8, 12, 30]
        self.use_multi_band = use_multi_band
        if use_multi_band:
            EEG_dict_new = {}
            for k, v in tqdm(EEG_dict.items(), desc='Filtering EEG'):
                eeg = EEG_dict[k]
                eeg = np.swapaxes(eeg, 1, 2)
                shape = eeg.shape
                eeg = np.reshape(eeg, (eeg.shape[0]*eeg.shape[1], -1))
                eeg = eeg.astype(np.float64)
                eeg_new_ls = []
                for i in range(len(self.band_limits)-1):
                    low, high = self.band_limits[i], self.band_limits[i + 1]
                    eeg_i = filter_data(eeg, 64, low, high,
                                        l_trans_bandwidth=1, h_trans_bandwidth=1,
                                        verbose='warning', filter_length=256, n_jobs=16)
                    eeg_i = np.reshape(eeg_i, shape)
                    eeg_i = np.swapaxes(eeg_i, 1, 2)
                    eeg_i = eeg_i.astype(np.float32)
                    eeg_new_ls.append(eeg_i)
                eeg_new = np.concatenate(eeg_new_ls, axis=2)
                # standardize
                eeg_new = (eeg_new - np.mean(eeg_new, axis=1, keepdims=True)) / (np.std(eeg_new, axis=1, keepdims=True) + 1e-8)
                EEG_dict_new[k] = eeg_new
            self.EEG_dict = EEG_dict_new
        else:
            self.EEG_dict = EEG_dict

        self.EEG_feat_index_dict = EEG_feat_index_dict
        self.Sub_id_dict = Sub_id_dict
        self.Stimulus_index_dict = Stimulus_index_dict
        self.id2dict = id2dict
        self.sub_names = sub_names
        self.feat_keys = feat_keys
        self.data_len = len(id2dict)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        t = self.id2dict[index]
        sub_idx, frame_idx = t[0], t[1]
        sub_name = self.sub_names[sub_idx]
        emb_id = self.EEG_feat_index_dict[sub_name][frame_idx]
        eeg = self.EEG_dict[sub_name][frame_idx]
        sub_id = self.Sub_id_dict[sub_name][frame_idx]
        sti_id = self.Stimulus_index_dict[sub_name][frame_idx]
        return eeg, emb_id, sub_id, sti_id


