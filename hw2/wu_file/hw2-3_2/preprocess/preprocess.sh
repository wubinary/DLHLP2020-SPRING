########################
## Make spectorgram data
data_root_dir=/media/D/DLHLP/hw2/Corpus/
h5py_path=vctk.h5
train_proportion=0.9

rm -f $h5py_path 
python3 make_dataset_vctk.py $data_root_dir $h5py_path $train_proportion

########################
## Make sample indexers
train_index_path=sample_segments/train_samples
valid_index_path=sample_segments/valid_samples
n_samples=500000
seg_len=128
speaker_used_path=speaker_used.txt 

# train indexer
python3 make_single_samples.py $h5py_path $train_index_path $n_samples $seg_len $speaker_used_path "train" 
# valid indexer
python3 make_single_samples.py $h5py_path $valid_index_path $n_samples $seg_len $speaker_used_path "test"


