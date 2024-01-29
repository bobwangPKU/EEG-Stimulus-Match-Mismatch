import os.path as op
import os

project_dir =  op.dirname(op.realpath(__file__))
derivatives_dir = os.path.join(project_dir, 'derivatives')
raw_dir = os.path.join(project_dir, 'rawdata')
data_dir = os.path.join(project_dir, 'data')

processed_eeg_dir = os.path.join(derivatives_dir, 'preprocessed_eeg')
processed_stimuli_dir = os.path.join(derivatives_dir, 'preprocessed_stimuli')

wav2vec_model_name = 'wav2vec2-large-xlsr-53-dutch'
wav2vec_model_path = os.path.join(project_dir, 'PM', wav2vec_model_name)

gpt_model_name = 'gpt2-small-dutch'
gpt_model_path = os.path.join(project_dir, 'PM', gpt_model_name)

# whisper_model_name = 'whisper-large-v3'
# whisper_model_path = op.join(project_dir, 'PM', whisper_model_name)