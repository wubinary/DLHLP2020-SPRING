Re-implementation of paper [Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1704.00849). on PyTorch


# Dependency
- Linux Ubuntu 16.04  
- Python 3.6 
    - PyTorch 0.4.0  
    - Numpy
    - Soundfile
    - PyWorld
<br/>


### Note:
1. Be sure to use create a virtual environment (using `conda` or `virtualenv`)
2. `soundfile` might require `sudo apt-get install` some codecs.  
<br/>
<br/>


# Usage
```bash
# source activate [your env]

bash download.sh

# install the required packages
pip install -r requirements.txt

# feature extraction
python analyzer.py \
--dir_to_wav dataset/vcc2016/wav \
--dir_to_bin dataset/vcc2016/bin 

# collect stats
python build.py \
--train_file_pattern "dataset/vcc2016/bin/Training Set/*/*.bin" \
--corpus_name vcc2016

# training
python main-vawgan.py --corpus_name vcc2016

# conversion
python convert-vawgan.py \
--corpus_name vcc2016 \
--src SF1 \
--trg TM3 \
--model_name [model_name].pt\
--file_pattern "./dataset/vcc2016/bin/Testing Set/{}/*.bin"
# Just an example; Please fill in `model_name`.
```

Description:  
1. Run `bash download.sh` to prepare the VCC2016 dataset.  
2. Run `analyzer.py` to extract features and write features into binary files. (This takes a few minutes.)  
3. Run `build.py` to collect stats, such as spectral extrema and pitch.  
4. To train a VAWGAN, for example, run  
5. You can find your models in `./model/[model_name].pt`  
6. To convert the voice, run  
7. You can find the converted wav files in `./logdir/output/[timestamp]`  

<br/>


# Dataset
Voice Conversion Challenge 2016 (VCC2016): [download page](https://datashare.is.ed.ac.uk/handle/10283/2211)  
<br/>

# Model  
 - [x] Conditional VAWGAN
<br/>
<br/>


# Folder Hierarchy
You only have to prepare the audio files in the following structure:
```
dataset
  vcc2016
    wav
      Training Set
      Testing Set
        SF1
        SF2
        ...
        TM3
```  
The speaker list will be built up automatically during training.  
<br/>



# Binary data format
The [WORLD vocdoer](https://github.com/mmorise/World) features and the speaker label are stored in binary format.  
Format:  
```
[[s1, s2, ..., s513, a1, ..., a513, f0, en, spk],
 [s1, s2, ..., s513, a1, ..., a513, f0, en, spk],
 ...,
 [s1, s2, ..., s513, a1, ..., a513, f0, en, spk]]
```
where   
`s_i` is (energy-normalized) spectral envelop magnitude (in log10) of the ith frequency bin,  
`a_i` is the corresponding "aperiodicity" feature,   
`f0` is the pitch (0 for unvoice frames),  
`en` is the energy,  
`spk` is the speaker index (0 - 9) and `s` is the `sp`.

Note:
  - The speaker identity `spk` was stored in `np.float32` but will be converted into `tf.int64` by the `reader` in `analysizer.py`.
  - I shouldn't have stored the speaker identity per frame;
    it was just for implementation simplicity. 

<br/>






