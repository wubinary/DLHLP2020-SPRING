# DLHLP_HW4-3

Baic part
-
To train a model:
```
bash train.sh
```
To generate the predition files for Kaggle:
```
bash predict.sh 
```
Note that in predict.sh you need to specify the model you want to use. For instance,
```
python ./examples/run_squad.py --model_name_or_path ./output/MY_MODEL --output_dir ./output/MY_MODEL
```


Lastly, in order to convert the output file predictions_.json to .csv (for Kaggle), you can use:
```
python process_ans.py <INPUT_FILE_NAME> <OUTPUT_FILE_NAME>
```
e.g.
```
python process_ans.py ./output/MY_MODEL/predictions_.json wulabaha.csv
```
Extra part (Optional)
-
If you can install apex successfully, you can use
```
python ./examples/run_squad.py --fp16
```
when training so as to acclerate and reduce the memory usage.
