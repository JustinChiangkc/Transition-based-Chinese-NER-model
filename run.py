 # -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
api = Api(app)
import jinja2
import json
import torch
import math
import os
import torch.nn as nn
from torch.optim import SGD, Adam, ASGD, Adagrad
from transition_framework_forIII import transition_forIII
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
parser.add_argument("--model_path", type=str,default="../save_model/save_68/model.pth")
parser.add_argument("--output", type=str, default=False)
parser.add_argument("--dataset", type=str, default='test')
parser.add_argument("--lang", type=str, default='zh_TW')

args = parser.parse_args()
print(args)

if args.output != False and args.dep == True:
    f_out = open(args.output, 'w')

print(f'{dt.now()} Loading embs')
word_char2id = get_id_map()
word_char_embedding = get_embedding()
print('vocab size:', word_char_embedding.size)
print('word embedding shape:', word_char_embedding.shape[0], word_char_embedding.shape[1])

pos2id = get_pos2id()
act2id = get_act2id()
pos_vocab_size = len(pos2id)
output_class = len(act2id)
meta_data = {'act2id': act2id,'output_class': output_class}
print('meta_data', meta_data)

######################## building models #########################

print(f'{dt.now()} Building model')
#hidden_size = load_hidden_size(args.model_path)
hidden_size = 2048
model = FFN(word_char_embedding, hidden_size, output_class, dropout_rate=0.4).cuda()
model.load_state_dict(torch.load(args.model_path))


def NERdecoder(sentence,debug = True):
    #input sentence as String
    NERresult = transition_forIII(model=model, input_is=sentence.split(), debug=True)
    NERresult_json = [{"position": e[0],"named_entity": e[1],"entity_type" : e[2],"probability" : e[3]} for e in NERresult]
    return NERresult_json



if args.output != False:
    f_out.close()


@app.route('/ner_tagger/', methods=['POST'])
def nerdecode():
    todo_id = request.query_string
    return jsonify({'NER':str(NERdecoder(todo_id))})


@app.route('/ner_tagger/ner_tag')
def ner_tag():
    return render_template('index.html')

@app.route('/ner_tagger/result',methods=['POST'])
def ner_result():
    rlt = NERdecoder(request.form['query'])
    return render_template('result.html',rlt=rlt)


# api.add_resource(NER, '/ner_tagger/<string:todo_id>')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
