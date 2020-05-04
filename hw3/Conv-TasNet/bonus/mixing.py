import os, json
import librosa
import numpy as np
import librosa.display

sound1, _ = librosa.load('bonus/data/s1/s1.wav', sr=8000)
sound2, _ = librosa.load('bonus/data/s2/s2.wav', sr=8000)
mix_sound = (sound1+sound2)/2
librosa.output.write_wav('bonus/data/mix/my_audio.wav', mix_sound, sr=8000)

def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

for speaker in ['mix', 's1', 's2']:
        preprocess_one_dir(os.path.join('bonus/data/', speaker),
                            'bonus/data/meta',
                            speaker,
                            sample_rate=8000)