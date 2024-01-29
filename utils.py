import config as cfg
import os.path as op
import soundfile as sf
import gzip
import numpy as np
import glob
import librosa
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2Model)
import torch
import pickle
from tqdm import tqdm
import os
from mne.filter import resample, filter_data
from sklearn.decomposition import PCA
from utils_gpt.GPT import GPT2
from utils_gpt.StimulusModel import LMFeatures1

def npz_to_wav():
    stimuli_dir = op.join(cfg.project_dir, 'stimuli')
    npz_stimuli_dir = op.join(stimuli_dir, 'eeg')
    wav_dir = op.join(stimuli_dir, 'stimuli_wav')
    if not op.exists(wav_dir):
        os.makedirs(wav_dir)
    # read .npz.gz of audiobook and postcast in npz_stimuli_dir and convert them to .wav, save in wav_dir
    for f in tqdm(os.listdir(npz_stimuli_dir)):
        if (f.endswith('.npz.gz') and not f.startswith('t')
                and not f.startswith('n') and 'noise' not in f and 'swn' not in f):
            npz_fname = op.join(npz_stimuli_dir, f)
            with gzip.open(npz_fname, "rb") as f_in:
                data = dict(np.load(f_in))
            wav_fname = op.join(wav_dir, f.replace('.npz.gz', '.wav'))
            try:
                audio = data['audio']
                fs = data['fs']
                sf.write(wav_fname, audio, fs)
            except:
                print(f'Error in {f}')
                continue

def audio_resample():
    audio_dir = op.join(cfg.project_dir, 'stimuli', 'stimuli_wav')
    audio_dir_16k = op.join(cfg.project_dir, 'stimuli', 'audio_16k')
    if not op.exists(audio_dir_16k):
        os.makedirs(audio_dir_16k)
    audio_files = glob.glob(op.join(audio_dir, '*.wav'))
    for audio_file in tqdm(audio_files):
        audio_file_16k = op.join(audio_dir_16k, op.basename(audio_file))
        if op.exists(audio_file_16k):
            continue
        y, sr = librosa.load(audio_file, 48000)
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write(audio_file_16k, y_16k, 16000)


def extract_wav2vec(layer=14, device='cuda:0'):
    emb_tag = f'-ly{layer}'
    emb_fname = op.join(cfg.data_dir, f'emb-{cfg.wav2vec_model_name}{emb_tag}.pkl')

    model_path = cfg.wav2vec_model_path
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    model = model.to(device)
    model = model.half()
    model.eval()
    with torch.no_grad():
        # for test size
        feat_ls = []
        audio_dir_16k = op.join(cfg.project_dir, 'stimuli', 'audio_16k')
        audio_files = glob.glob(op.join(audio_dir_16k, '*.wav'))
        audio_files.sort()
        audio_names = [op.basename(audio_file) for audio_file in audio_files]
        print(f'extracting feat wav2vec2 layer {layer}')
        for wav_fname in tqdm(audio_files):
            feat_st_ls = []
            wav, sr = librosa.load(wav_fname, sr=16000)
            for sp_st in tqdm(range(0, len(wav), 5*int(sr))):
                wav_i = wav[sp_st:sp_st+5*sr+100]
                if len(wav_i) < 500:
                    continue
                wav_input = feature_extractor(wav_i, return_tensors='pt', sampling_rate=sr)
                input_values = wav_input.input_values.half().to(device)
                o = model(input_values, output_hidden_states=True)
                feat_st_ls.append(o[2][layer].detach().squeeze().cpu().numpy())
            feat_st = np.concatenate(feat_st_ls)
            feat_ls.append(feat_st)
        emb_dict = {audio_names[i]: feat_ls[i] for i in range(len(audio_names))}

        data = {}
        data['emb_dict'] = emb_dict
        with open(emb_fname, 'wb') as f:
            pickle.dump(data, f)



def pca_wav2vec(layer=14, pca_dim=64):
    emb_tag = f'-ly{layer}'
    with open(op.join(cfg.data_dir, f'emb-{cfg.wav2vec_model_name}{emb_tag}.pkl'), 'rb') as f:
        data = pickle.load(f)
    emb_dict = data['emb_dict']
    emb_dict_keys = list(emb_dict.keys())
    emb_dict_vals = np.concatenate([emb_dict[key] for key in emb_dict_keys], axis=0)
    emb_dict_len = [emb_dict[key].shape[0] for key in emb_dict.keys()]


    print(f'PCA feat wav2vec2 layer {layer}')
    pca = PCA(n_components=pca_dim)
    emb_dict_vals_pca = pca.fit_transform(emb_dict_vals)
    emb_dict_vals_pca_split = np.split(emb_dict_vals_pca, np.cumsum(emb_dict_len)[:-1])
    emb_dict_pca = {emb_dict_keys[i]: emb_dict_vals_pca_split[i] for i in range(len(emb_dict_keys))}

    data = {}
    data['emb_dict'] = emb_dict_pca
    data['pca'] = pca
    pca_tag = f'pca{pca_dim}' if pca_dim != 64 else 'pca'
    with open(op.join(cfg.data_dir, f'emb-{cfg.wav2vec_model_name}{emb_tag}-{pca_tag}.pkl'), 'wb') as f:
        pickle.dump(data, f)

    data = {}
    data['pca'] = pca
    with open(op.join(cfg.data_dir, f'emb-{cfg.wav2vec_model_name}{emb_tag}-pcaw{pca_dim}.pkl'), 'wb') as f:
        pickle.dump(data, f)

def resample_align_wav2vec(layer=14, pca_dim=64):
    emb_tag = f'-ly{layer}'
    pca_tag = f'pca{pca_dim}' if pca_dim != 64 else 'pca'
    feat_name = f'wav2vec{layer}{pca_tag}'
    with open(op.join(cfg.data_dir, f'emb-{cfg.wav2vec_model_name}{emb_tag}-{pca_tag}.pkl'), 'rb') as f:
        data = pickle.load(f)


    emb_dict = data['emb_dict']
    emb_dict_keys = list(emb_dict.keys())
    save_dir = op.join(cfg.derivatives_dir, 'preprocessed_stimuli')
    env_dir = cfg.processed_stimuli_dir
    env_files = glob.glob(op.join(env_dir, '*_-_envelope.npy'))
    print(f'resampling feat wav2vec2 layer {layer} for trainset')
    for env_f in tqdm(env_files):
        env = np.load(env_f)
        env_name = op.basename(env_f)
        audio_name = env_name.replace('_-_envelope.npy', '.wav')
        if audio_name not in emb_dict_keys:
            continue
        emb = emb_dict[audio_name]
        emb = emb.astype(np.float64)
        emb_res = resample(emb.T, len(env)/len(emb), 1.0)
        emb_res = emb_res.T
        emb_res_2 = resample(emb.T, 64/50, 1.0).T
        len_diff = len(emb_res_2) - len(emb_res)
        print(f'len_diff: {len_diff}')
        #  padding or truncating to ensure the same length
        if len(emb_res) > len(env):
            emb_res = emb_res[:len(env), :]
        elif len(emb_res) < len(env):
            emb_res = np.concatenate((emb_res, np.zeros((len(env)-len(emb_res), emb_res.shape[1]))), axis=0)
        emb_res = emb_res.astype(np.float32)
        emb_name = op.basename(env_f).replace('envelope', f'{feat_name}')
        emb_f = op.join(save_dir, emb_name)
        np.save(emb_f, emb_res)

def whisper_time_align(device='cuda:0'):
    import stable_whisper
    audio_dir_16k = op.join(cfg.project_dir, 'stimuli', 'audio_16k')
    save_dir = op.join(cfg.project_dir, 'stimuli', 'audio_16k_aligned')
    if not op.exists(save_dir):
        os.makedirs(save_dir)
    audio_files = glob.glob(op.join(audio_dir_16k, '*.wav'))
    audio_files.sort()
    model = stable_whisper.load_model('large-v3', device=device)
    for wav_fname in tqdm(audio_files):
        result = model.transcribe(wav_fname,
                                  logprob_threshold=-100,
                                  no_speech_threshold=0.9,
                                  compression_ratio_threshold=10,
                                  condition_on_previous_text=False,
                                  word_timestamps=True)
        wav_name = op.basename(wav_fname)
        save_vtt_fname = op.join(save_dir, wav_name.replace('.wav', '.vtt'))
        result.to_srt_vtt(save_vtt_fname, segment_level=False, word_level=True)

def extract_GPT(layer=9, context_words=5, device='cuda:0'):
    gpt = GPT2(path=cfg.gpt_model_path, device=device)
    features = LMFeatures1(model=gpt, layer=layer, context_words=context_words)
    aligned_dir = op.join(cfg.project_dir, 'stimuli', 'audio_16k_aligned')
    vtt_files = glob.glob(op.join(aligned_dir, '*.vtt'))
    vtt_files.sort()
    emb_dict = {}
    onset_dict = {}
    offset_dict = {}
    for vtt_file in tqdm(vtt_files):
        audio_name = op.basename(vtt_file).replace('.vtt', '.wav')
        with open(vtt_file, 'r') as f:
            lines = f.readlines()
        time_info = lines[2::3]
        onset = [t.split('-->')[0] for t in time_info]
        onset = [float(t.split(':')[0])*3600 + float(t.split(':')[1])*60 + float(t.split(':')[2]) for t in onset]
        offset = [t.split('-->')[1] for t in time_info]
        offset = [float(t.split(':')[0])*3600 + float(t.split(':')[1])*60 + float(t.split(':')[2]) for t in offset]
        word = lines[3::3]
        # remove punctuation if any
        punctuation = ['.', ',', '?', '!', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '-', '—', '–']
        word = [w.strip() for w in word]
        word_new = []
        for w in word:
            for p in punctuation:
                w = w.replace(p, '')
            word_new.append(w)
        word_vecs = features.make_stim(word_new)
        if np.isnan(word_vecs).any():
            nan_num = len(np.argwhere(np.isnan(word_vecs))) // 768
            word_vecs[np.isnan(word_vecs)]= np.repeat(np.nanmean(word_vecs, axis=0, keepdims=True), nan_num, axis=0).reshape(-1)

        emb_dict[audio_name] = word_vecs
        onset_dict[audio_name] = onset
        offset_dict[audio_name] = offset
    data = {}
    data['emb_dict'] = emb_dict
    data['onset_dict'] = onset_dict
    data['offset_dict'] = offset_dict
    emb_tag = f'-ly{layer}-cw{context_words}'
    with open(op.join(cfg.data_dir, f'emb-{cfg.gpt_model_name}{emb_tag}.pkl'), 'wb') as f:
        pickle.dump(data, f)

def pca_GPT(layer=9, context_words=5, pca_dim=4):
    emb_tag = f'-ly{layer}-cw{context_words}'
    with open(op.join(cfg.data_dir, f'emb-{cfg.gpt_model_name}{emb_tag}.pkl'), 'rb') as f:
        data = pickle.load(f)
    onset_dict = data['onset_dict']
    offset_dict = data['offset_dict']
    emb_dict = data['emb_dict']
    emb_dict_keys = list(emb_dict.keys())
    emb_dict_vals = np.concatenate([emb_dict[key] for key in emb_dict_keys], axis=0)
    emb_dict_len = [emb_dict[key].shape[0] for key in emb_dict.keys()]

    pca = PCA(n_components=pca_dim)
    emb_dict_vals_pca = pca.fit_transform(emb_dict_vals)
    emb_dict_vals_pca_split = np.split(emb_dict_vals_pca, np.cumsum(emb_dict_len)[:-1])


    emb_dict_pca = {k:v for k,v in zip(emb_dict_keys, emb_dict_vals_pca_split)}
    data = {}
    data['emb_dict'] = emb_dict_pca
    data['pca'] = pca
    data['onset_dict'] = onset_dict
    data['offset_dict'] = offset_dict
    with open(op.join(cfg.data_dir, f'emb-{cfg.gpt_model_name}{emb_tag}-pca.pkl'), 'wb') as f:
        pickle.dump(data, f)

    data = {}
    data['pca'] = pca
    with open(op.join(cfg.data_dir, f'emb-{cfg.gpt_model_name}{emb_tag}-pcaw.pkl'), 'wb') as f:
        pickle.dump(data, f)

def resample_align_gpt(layer=9, context_words=5, fs=64):
    emb_tag = f'-ly{layer}-cw{context_words}'
    feat_name = f'gpt{layer}cw{context_words}pca'
    with open(op.join(cfg.data_dir, f'emb-{cfg.gpt_model_name}{emb_tag}-pca.pkl'), 'rb') as f:
        data = pickle.load(f)
    onset_dict = data['onset_dict']
    offset_dict = data['offset_dict']
    emb_dict = data['emb_dict']
    emb_dict['audiobook_1_1_shifted.wav'] = emb_dict['audiobook_1_1.wav'].copy()
    emb_dict['audiobook_1_2_shifted.wav'] = emb_dict['audiobook_1_2.wav'].copy()
    onset_dict['audiobook_1_1_shifted.wav'] = onset_dict['audiobook_1_1.wav'].copy()
    onset_dict['audiobook_1_2_shifted.wav'] = onset_dict['audiobook_1_2.wav'].copy()
    offset_dict['audiobook_1_1_shifted.wav'] = offset_dict['audiobook_1_1.wav'].copy()
    offset_dict['audiobook_1_2_shifted.wav'] = offset_dict['audiobook_1_2.wav'].copy()
    emb_dict_keys = list(emb_dict.keys())
    save_dir = op.join(cfg.derivatives_dir, 'preprocessed_stimuli')
    env_dir = op.join(cfg.derivatives_dir, 'preprocessed_stimuli')
    env_files = glob.glob(op.join(env_dir, '*_-_envelope.npy'))
    for env_f in tqdm(env_files):
        env = np.load(env_f)
        env_name = op.basename(env_f)
        audio_name = env_name.replace('_-_envelope.npy', '.wav')
        if audio_name not in emb_dict_keys:
            continue
        emb = emb_dict[audio_name]
        emb = emb.astype(np.float64)
        onset = onset_dict[audio_name]
        offset = offset_dict[audio_name]
        emb_cont = np.zeros((len(env), emb.shape[1]))
        for i in range(len(onset)):
            onset_i = round(onset[i]*fs)
            offset_i = round(offset[i]*fs)
            emb_cont[onset_i:offset_i, :] = emb[i]
        emb_cont_filterd = filter_data(emb_cont.T, fs, 0, 4).T
        emb_name = op.basename(env_f).replace('envelope', f'{feat_name}')
        emb_f = op.join(save_dir, emb_name)
        np.save(emb_f, emb_cont_filterd)


if __name__ == '__main__':
    '''
    convert .npz.gz to .wav, and resample to 16k
    '''
    npz_to_wav()
    audio_resample()
    '''
    extract wav2vec feature and PCA
    '''
    layer = 14
    extract_wav2vec(layer=layer, device='cuda:1')
    pca_wav2vec(layer=layer, pca_dim=64)
    resample_align_wav2vec(layer=layer, pca_dim=64)
    '''
    extract gpt feature and PCA
    '''
    whisper_time_align(device='cuda:0') # get words and their onset and offset from wav
    extract_GPT(layer=9, context_words=5)
    pca_GPT(layer=9, context_words=5)
    resample_align_gpt(layer=9, context_words=5, fs=64)
