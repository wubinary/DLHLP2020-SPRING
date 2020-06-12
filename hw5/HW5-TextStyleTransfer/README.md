# Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation

This folder contains the code for the paper [《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》](https://arxiv.org/abs/1905.05621)



## Requirements

1. pip packages
    - pytorch>=0.4.0
    - torchtext>=0.4.0
    - nltk
    - fasttext==0.8.3
    - pypi-kenlm
    - tqdm
    - wandb
    
2. LM for evaluator
    - Download https://drive.google.com/open?id=1pklyWxzNPPxnKNy_TmA8h_tmGmiZttPN
    - Put `ppl_yelp.binary` in the folder `evaluator`

3. nltk setup
    - Run the following command to download (or double-check) nltk tokenizer dependency:
    ```
    python -c "import nltk; nltk.download('punkt')"
    ```

## Usage

The hyperparameters for the Style Transformer can be found in `main.py` or with `python main.py -h`.

- To train:
```shell
python main.py --do_train
```
- To inference on test set:
```shell
SAVE=./save/Feb15141010/ckpts/2000 # just example
python main.py --do_test \
-test_out ./submission.txt \
-preload_F ${SAVE}_F.pth \
-preload_D ${SAVE}_D.pth
```
You can modify other parameters to suit your need.

- To run hw5-2

```shell
SAVE=./save/Feb15203331/ckpts/ #just example
STEP=1300
python main.py --part2 \
--part2_model_dir $SAVE \
--part2_step $STEP
```
