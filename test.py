import json
import torch
import math
import os
import torch.nn as nn
from torch.optim import SGD, Adam, ASGD, Adagrad
from transition_framework import transition
from data_utils import *
from evaluation import *
from utility import *
from datetime import datetime as dt
from model import FFN
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_hidden_size(model_path):
    # model_path is full model_path, we are going to find the args.txt file in same dir
    model_dir = os.path.dirname(model_path)
    args_path = os.path.join(model_dir, 'args.txt')
    with open(args_path, 'r') as f:
        for line in f:
            if line.split(',')[0] == 'hidden_size':
                return int(line.strip().split(',')[1])
        else:
            raise ValueError(f"No hidden_size mentioned in {args_path}")

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--debug", type=str2bool, nargs='?', const=False, default=False)
parser.add_argument("--model_path", type=str)
parser.add_argument("--output", type=str, default=False)
parser.add_argument("--dataset", type=str, default='test')
parser.add_argument("--seg", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--pos", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--dep", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--lang", type=str, default='zh_TW')
parser.add_argument("--testset_path", type=str, default="../data/insurance_dataset/test1")

args = parser.parse_args()
print(args)

if args.output != False and args.dep == True:
    f_out = open(args.output, 'w')

DATA = {}
DATA['test'] = args.testset_path

print(f'{dt.now()} Loading embs')
word_char2id = get_id_map()
word_char_embedding = get_embedding()
print('vocab size:', word_char_embedding.size)
print('word embedding shape:', word_char_embedding.shape[0], word_char_embedding.shape[1])

pos2id = get_pos2id()
act2id = get_act2id()
pos_vocab_size = len(pos2id)
output_class = len(act2id)
meta_data = {'pos2id': pos2id, 'act2id': act2id,  
            'pos_vocab_size': pos_vocab_size, 'output_class': output_class}
print('meta_data', meta_data)

######################## building models #########################

print(f'{dt.now()} Building model')
#hidden_size = load_hidden_size(args.model_path)
hidden_size = 2048
model = FFN(word_char_embedding, hidden_size, output_class, dropout_rate=0.4).cuda()
model.load_state_dict(torch.load(args.model_path))

data_path = DATA[args.dataset]
dataset = Dataset(data_path, word_char2id)
dataloader = get_one_dataloader(dataset)

ner_f1_caculator = F1_Calculator()


for batch, (chars, gold_ner) in enumerate(dataloader):
    if args.debug == True:
        print()
        print('chars', chars)
        print('chars[0]', chars[0])
        print('gold_ner[0]', gold_ner[0])
        # print('gold_pos', gold_pos[0])
        # print('gold_dep', gold_dep[0])
        input('------------- input to continue ---------------')
    if args.output != False:
        f_out.write(f'{batch} {chars[0]}\n')
    pred_ner = transition(model=model, input_is=chars[0].split(), debug=args.debug)
    if args.output != False :
        for ner in pred_ner:
            f_out.write(f'{str(ner)}\n')
        f_out.write('\n')
    if args.debug == True:
        print('gold_ner[0]:', gold_ner[0])
        print('pred_ner:', pred_ner)


        input('------------- input to continue ---------------')
    ner_f1_caculator.update(eval_ner(pred_ner, gold_ner))


    ner_f1 = ner_f1_caculator()


    print(f'\r{dt.now()} {args.dataset}:{batch} ner F1:{ner_f1:.4f}', end='')



if args.output != False:
    f_out.close()

