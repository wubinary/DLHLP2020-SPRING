
for layer in {0..11}
do
    for task in {parse-distance,parse-depth}
    do
        python structural-probes/experiment_yaml.py --model_layer $layer --task_name $task 
        python structural-probes/run_experiment.py example/config/ctb/pad_ctb-BERTbase.yaml 
    done
done