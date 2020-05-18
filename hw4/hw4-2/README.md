# DLHLP HW4-2


## Step 1
Clone the repository below:
```bash=
git clone https://github.com/Sologa/structural-probes.git
```
## Step 2
Download dataset and put it under `data/ctb`.
## Step 3
Perform preprocessing:
```bash=
bash scripts/gen_ctb.sh
```
## Step 4
Perform training:
```bash=
bash scripts/call_exp.sh
```
## Step 5
Run demo.
```bash=
bash scripts/seg.sh CONFIG.JSON_OF_YOUR_4-1_SUMISSION MY_CWS_BERT.PT_OF_YOUR_4-1_SUMISSION
```
Put the transformer directory containing your modeling_bert.py used in cws of 4-1 to assist segmentation.
Finally, you will have your .tikz in seg_results/

## Step 6
Plot several dependency graphs given examples from dev set. The tikz file for visualization can be found after training at `example/results/[checkpoint_name]/dev.tikz`. You can use online service like Overleaf to complile the tikz file. An example could be found here: https://www.overleaf.com/read/bqcszyjrgnby.

## Step 7
For bonus task (learning to *rank* instead of directly regressing), complete the TODO loss function in [structural-probes/loss.py](structural-probes/loss.py). Instead of using squared Euclidean distance as in the paper, you can also take just the Euclidean distance and perform rank loss on it by adding a probe class taking the square root of the inner product in [structural-probes/probe.py](structural-probes/probe.py), and register the probe at [structural-probes/run_experiment.py](structural-probes/run_experiment.py).

