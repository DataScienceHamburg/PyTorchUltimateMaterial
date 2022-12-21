#%% package import
import torchaudio
from plot_audio  import plot_specgram
import os
import random
# %%
wav_path = 'data/set_a'
wav_filenames = os.listdir(wav_path)
random.shuffle(wav_filenames)

# %%
ALLOWED_CLASSES = ['normal', 'murmur', 'extrahls', 'artifact']
for f in wav_filenames:
    class_type = f.split('_')[0]
    f_index = wav_filenames.index(f)
    # if file position is 0-139 then train folder, else test
    target_path = 'train' if f_index < 140 else 'test'
    class_path = f"{target_path}/{class_type}"
    file_path = f"{wav_path}/{f}"
    f_basename = os.path.basename(f)
    f_basename_wo_ext = os.path.splitext(f_basename)[0]
    target_file_path = f"{class_path}/{f_basename_wo_ext}.png"
    if (class_type in ALLOWED_CLASSES):
        # create folder if necessary
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        # extract class type from file, e.g. 
        data_waveform, sr = torchaudio.load(file_path)
        # create spectrogram and save it
        
        plot_specgram(waveform=data_waveform, sample_rate=sr, file_path=target_file_path)
        
#%%
