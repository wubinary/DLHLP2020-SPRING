#!/bin/bash

mkdir -p ckpt/asr_dlhlp_ctc_sd0
if [ ! -f "ckpt/asr_dlhlp_ctc_sd0/best_att.pth" ]; then
	wget http://lab.wubinray.com/model/DLHLP2020-SPRING/hw1/asr_dlhlp_ctc_sd0/best_att.pth -P ckpt/asr_dlhlp_ctc_sd0/
else
	printf "asr model exist\n"
fi

mkdir -p ckpt/lm_dlhlp_sd0
if [ ! -f "ckpt/lm_dlhlp_sd0/best_ppx.pth" ]; then
	wget http://lab.wubinray.com/model/DLHLP2020-SPRING/hw1/lm_dlhlp_sd0/best_ppx.pth -P ckpt/lm_dlhlp_sd0/
else
	printf "lm model exist\n"
fi

part1=`dirname "$1"`
part2=`basename "$1"`

## asr
sed -i "s+/media/D/DLHLP+${part1}+g" config/libri/asr_dlhlp_ctc.yaml
## lm
sed -i "s+/media/D/DLHLP+${part1}+g" config/libri/lm_dlhlp.yaml

NJOBS=6
#make test_CTC
python3 reproduce.py --config config/libri/decode_dlhlp_ctc.yaml --test --njobs $NJOBS

make ANS_CSV_PATH=$2 submit_reproduce

