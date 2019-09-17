import pickle
import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import datetime as dt
from function import *
import itertools


print(f'{dt.now()} load lexical2id')
with open('../data/embedding_map_CG.pkl', 'rb') as f:
    lexical2id = pickle.load(f)

print(f'{dt.now()} load lexical_embedding' )
with open('../data/embedding_matrix_CG.pkl', 'rb') as f:
    lexical_embedding = pickle.load(f)

print(f'{dt.now()} load pos2id')
with open('../data/pos2id.json', 'r') as f:
    pos2id = json.load(f)


print(f'{dt.now()} load act2id')
with open('../data/act2id_NER.json', 'r') as f:
    act2id = json.load(f)


print(f'vocab size (include UNK and PAD): {len(lexical2id)}')

def get_embedding():
    return lexical_embedding

def get_id_map():
    return lexical2id

def get_pos2id():
    return pos2id

def get_act2id():
    return act2id


def get_one_dataloader(dataset, num_workers=8):
    dataloader = DataLoader(dataset,
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  collate_fn=eval_collate_fn, 
                                                  pin_memory=True,  
                                                  num_workers=num_workers)
    return dataloader
def get_data_loader(train_dataset, dev_dataset, test_dataset, batch_size, num_workers):
    train_dataloader = DataLoader(train_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True, 
                                                  collate_fn=collate_fn,
                                                  pin_memory=True,  
                                                  num_workers=num_workers)
    train_eval_dataloader = DataLoader(train_dataset, 
                                                  batch_size=1, 
                                                  shuffle=True, 
                                                  collate_fn=eval_collate_fn,
                                                  pin_memory=True,  
                                                  num_workers=num_workers)
    dev_dataloader = DataLoader(dev_dataset,
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  collate_fn=eval_collate_fn,
                                                  pin_memory=True,   
                                                  num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset,
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  collate_fn=eval_collate_fn, 
                                                  pin_memory=True,  
                                                  num_workers=num_workers)
    return train_dataloader, train_eval_dataloader, dev_dataloader, test_dataloader

class Dataset(Dataset):
    def __init__(self, data_dir, word2id):
        super(Dataset).__init__()
        self.data_paths = os.listdir(data_dir)
        self.data_paths = sorted(self.data_paths)
        self.data_paths = [os.path.join(data_dir, x) for x in self.data_paths]

    def __getitem__(self, index):
        with open(self.data_paths[index], 'r') as f:
            data = json.load(f)
        return data
    def __len__(self):
        return len(self.data_paths)

def make_act2id(action, act2id):
    return act2id[action]

def eval_collate_fn(datas):
    seg_sentence, gold_ners = [], []
    # numbers of action from each sentence
    length_in_batch = [len(x) for x in datas]
    for data in datas:
        seg_sentence.append(data['seg_sentence'])
        gold_ner = data['gold_ner']
        gold_ners.append(gold_ner)

    # wfcs: cannot transform to list
    return seg_sentence, gold_ners

def collate_fn(datas):
    '''
    1. trun text to feature used in model (state feature)
    2. merge minibatch
    
    wfw: word feature in word id
    wfc: word feature in char id
    char_mask: char mask
    pfp: pos feature in pos id
    cfc: char feature in char id
    lenq0: lenq0

    Args:
        datas: list of list of states, lenght == sentence_num == batch_num
    '''
    wfws, wfcs, char_masks, offsets, seg_sents, lenq0s = [], [], [], [], [], []

    actions = []
    # numbers of action from each sentence
    length_in_batch = [len(x) for x in datas]
    for data in datas:
        # data is a list of state
        seg_sentence = data['seg_sentence']
        configurations = data['configurations']
        for config in configurations: 
            stack, buffer = config['stack'], config['buffer']
            action = config['action']
            feature_in_text = turn_feature_in_text(stack, 
                                                   buffer) 

            feature = feature2id(feature_in_text, lexical2id)

            word_feature_in_wordid, word_feature_in_charid,\
                char_mask, lenq0 = feature
            action = make_act2id(action, act2id)

            wfws.append(word_feature_in_wordid)

            a_wfc, a_offset = [], []
            for wfc in word_feature_in_charid:
                a_wfc += wfc
                a_offset.append(len(wfc))

            wfcs += a_wfc
            char_masks.append(char_mask)
            offsets += a_offset

            lenq0s.append(lenq0)

            actions.append(action)
        seg_sents.append(seg_sentence)
   

    wfws = torch.LongTensor(wfws)

    wfcs = torch.LongTensor(wfcs)
    char_masks = torch.LongTensor(char_masks)
    offsets = (list(itertools.accumulate(offsets, lambda x, y : x+y)))
    offsets.insert(0, 0)
    offsets = offsets[:-1]
    offsets = torch.LongTensor(offsets)

    lenq0s = torch.LongTensor(lenq0s)
    actions = torch.LongTensor(actions)

    # wfcs: cannot transform to list
    return wfws, wfcs, char_masks, offsets, lenq0s, actions, length_in_batch, seg_sents
