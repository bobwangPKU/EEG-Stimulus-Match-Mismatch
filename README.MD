# Code to implement the model of No.2 in Task 1 of the Auditory EEG Challenge (ICASSP 2024)
The paper is available at: https://arxiv.org/abs/2401.04964. 
The official website of the challenge is: https://exporl.github.io/auditory-eeg-challenge-2024/.
## Requirements
- Python 3.9
- numpy 1.21.6
- mne 1.3.1
- pickle
- librosa 0.9.2
- soundfile 0.12.1
- torch 1.12.0
- scikit-learn 1.2.1
- transformers 4.33.3
- tensorboard 2.10.0
- tqdm 4.64.1
## Dataset
The dataset used here is SparrKULee, which can be downloaded from the
[KU Leuven RDR](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND). 
The processed EEG provided by the dataset was used.
## Pretrained Model
### Wav2vec 2.0
Self-supervised speech model wav2vec 2.0 was used to extract speech representation. 
In practice, we used the [wav2vec2-large-xlsr-53-dutch](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch), 
which has been pretrained on 56,000 hours of speech from 53 different languages and fine-tuned on Dutch.
### GPT-2
Pretrained language model GPT-2 was used to extract contextual text embedding. 
In practice, we used the [gpt2-small-dutch](https://huggingface.co/GroNLP/gpt2-small-dutch), which has been pretrained on English text and adapted to Dutch.

## How to run
### Prepare EEG data
Download SparrKULee dataset, and put the `derivatives` and `stimuli` folder in the root directory of the project.
### Prepare Pretrained Model
Download the pretrained model of wav2vec 2.0 and GPT-2, and put them in the `PM` folder.
### Prepare features
Run `python utils.py` to extract features from the speech.
### Train
Run `python train_models.py` to train the model.

To train the model with different hyperparameters, you can modify the `model_config.py` file.

For example, to train the model with different features, 
you can modify the `feature_name` parameter as `wav2vec14pca_gpt9cw5pca` to use wav2vec 2.0 feature and gpt feature (Names of each feature are concatenated by `_`). 

## Acknowledgement
The code to extract GPT-2 feature is modified from [this repo](https://github.com/HuthLab/semantic-decoding).
