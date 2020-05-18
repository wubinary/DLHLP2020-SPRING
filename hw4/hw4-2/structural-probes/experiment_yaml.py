
import yaml
from argparse import ArgumentParser


argp = ArgumentParser()
argp.add_argument('--model_layer', default=0, help='0 ~ 11')
argp.add_argument('--task_name', default=0, help='parse-distance or parse-depth')
args = argp.parse_args()


file = yaml.load(open('example/config/ctb/pad_ctb-BERTbase.yaml'))
file['model']['model_layer'] = int(args.model_layer)
# file['probe']['depth_params_path'] = '../example/data/bertbertchinese_{}-depth-probe.params'.format(args.model_layer)
# file['probe']['distance_params_path'] = '../example/data/bertchinese_{}-distance-probe.params'.format(args.model_layer)
file['probe']['task_name'] = args.task_name
file['probe']['params_path'] = 'predictor_{}_{}.params'.format(args.model_layer, args.task_name)

file['probe']['maximum_rank'] = 128
file['dataset']['batch_size'] = 10
file['probe_training']['epochs'] = 30

if args.task_name == 'parse-depth':
    file['probe']['task_signature'] = 'word'
    file['reporting']['reporting_methods'] = ['spearmanr', 'root_acc']
else:
    file['probe']['task_signature'] = 'word_pair'
    file['reporting']['reporting_methods'] = ['spearmanr', 'uuas']

yaml.dump(file, open('example/config/ctb/pad_ctb-BERTbase.yaml', 'w'))
