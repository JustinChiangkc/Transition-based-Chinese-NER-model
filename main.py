import json
import os
import torch
import math
import torch.nn as nn
from torch.optim import SGD, Adam, ASGD, Adagrad, RMSprop
from transition_framework import transition
from data_utils import get_embedding, get_id_map, get_act2id, Dataset, get_data_loader
from evaluation import eval_dep_labeled, eval_dep_unlabeled, eval_ner, F1_Calculator, cal_loss
from utility import find_save_dir, save_model_with_result, save_test_result
from datetime import datetime as dt
from model import FFN
import argparse

######################## parsing arguments ########################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=8, help='thread number for dataloader')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--optimizer', type=str, default='adagrad')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--label', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--cuda', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--earlystop', type=float, default=50)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--l2', type=float, default=0.000001)
parser.add_argument('--lang', type=str, default='zh_TW', help='determing language to use, options are zh_TW or zh_CN')
parser.add_argument('--conductTrain', type=bool, default=False)
parser.add_argument('--data_path', type=str, default="../data/insurance_dataset/")


args = parser.parse_args()
print(args)


save_dir_path = find_save_dir('save_model', 'save')

with open(os.path.join(save_dir_path, 'args.txt'), 'w') as f:
    for key, value in vars(args).items():
        #print(f'{key},{str(value)}\n')
        f.write(f'{key},{str(value)}\n')
TRAIN_DATA = args.data_path + "train"
VALID_DATA = args.data_path + "dev"
TEST_DATA = args.data_path + "test"


######################## reading meta datas ########################

print(f'{dt.now()} Loading embs')
word_char2id = get_id_map()
word_char_embedding = get_embedding()
print('vocab size:', word_char_embedding.size)
print('word embedding shape:', word_char_embedding.shape[0], word_char_embedding.shape[1])

# pos2id = get_pos2id()
act2id = get_act2id()
# pos_vocab_size = len(pos2id)
output_class = len(act2id)
meta_data = {'act2id': act2id, 'output_class': output_class}
print('meta_data', meta_data)

######################## building models #########################

print(f'{dt.now()} Building model')
if args.cuda:
    model = FFN(word_char_embedding, args.hidden_size, output_class, args.dropout_rate).cuda()
else:
    model = FFN(word_char_embedding, args.hidden_size, output_class, arge.dropout_rate)

######################## reading meta datas ########################

print(f'{dt.now()} Loading datas')
train_dataset = Dataset(TRAIN_DATA, word_char2id)
valid_dataset = Dataset(VALID_DATA, word_char2id)
test_dataset = Dataset(TEST_DATA, word_char2id)
print(f'{dt.now()} Building dataloader')
train_dataloader, train_eval_dataloader, valid_dataloader, test_dataloader \
        = get_data_loader(train_dataset, valid_dataset, test_dataset, args.batch_size, args.num_workers)


if args.optimizer.lower() == 'adagrad':
    optimizer = Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)
elif args.optimizer.lower() == 'adam':
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)
else:
    raise(ValueError(f'Unknown optimizer, given is {args.optimizer}'))

loss_function = nn.CrossEntropyLoss()

####################### Start Training #############################


print(f'{dt.now()} Start Training')
earlystop_counter, earlystop_flag = 0, 0.0
for epoch_num in range(args.epoch):
    total_loss = 0
    loss_count, valid_loss_count = 0, 0
    ner_f1_caculator = F1_Calculator()
    for batch, (wfws, wfcs, char_masks, offsets, lenq0s, gold_actions, length_in_batch,\
            sentences) in enumerate(train_dataloader):
        if args.cuda:
            wfws = wfws.cuda()
            wfcs = wfcs.cuda()
            char_masks = char_masks.cuda()
            offsets = offsets.cuda()
            lenq0s = lenq0s.cuda()
            gold_actions = gold_actions.cuda()
        model = model.train()
        model.zero_grad(); optimizer.zero_grad()#?
        pred_actions = model(wfws, wfcs, char_masks, offsets, lenq0s)
        loss = cal_loss(pred_actions, gold_actions, loss_function)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 0.25)
        optimizer.step()
    
        total_loss += loss.item()
        loss_count += 1
        loss = total_loss / loss_count
        print(f'\r{dt.now()} Epoch: {epoch_num} Train:{batch} Loss:{loss:.4f}', end='')
    print()

    print("Sameple 200 sentence to see the accuracy in training set")
    # sameple 200 sentence to see the accuracy in training set
    for batch, (chars, gold_ner) in enumerate(train_eval_dataloader):
        if batch >= 200:
            break
        pred_ner = transition(model=model, input_is=chars[0].split(), args=args, debug=False)
    
        ner_f1_caculator.update(eval_ner(pred_ner, gold_ner))
        ner_f1 = ner_f1_caculator()

        print(f'\r{dt.now()} Epoch: {epoch_num} Train:{batch} ner F1:{ner_f1:.4f}', end='')
    print()
   
    ########################### Validation #################################
    ner_f1_caculator = F1_Calculator()
    for batch, (chars, gold_ner) in enumerate(valid_dataloader):
        pred_ner = transition(model=model, input_is=chars[0].split(), args=args)

        ner_f1_caculator.update(eval_ner(pred_ner, gold_ner))
        ner_f1 = ner_f1_caculator()


        print(f'\r{dt.now()} Epoch: {epoch_num} Valid:{batch} ner F1:{ner_f1:.4f}', end='')
    print()
    if earlystop_flag < ner_f1:
        earlystop_flag  = ner_f1
        save_model_with_result(model, save_dir_path, ner_f1, args)
        earlystop_counter = 0
    else:
        earlystop_counter += 1
    if earlystop_counter >= args.earlystop:
        print('\nearlystop!!')
        print('model save at', save_dir_path)
        break
    print()


#################################### Test #####################################    
# In testing, we load model from previous save best
print("Start test")
if args.cuda:
    model = FFN(word_char_embedding, args.hidden_size, output_class, args.dropout_rate).cuda()
else:
    model = FFN(word_char_embedding, args.hidden_size, output_class, arge.dropout_rate)
model.load_state_dict(torch.load(os.path.join(save_dir_path, 'model.pth')))
ner_f1_caculator = F1_Calculator()
for batch, (chars, gold_ner) in enumerate(test_dataloader):
    pred_ner = transition(model=model, input_is=chars[0].split(), args=args)


    ner_f1_caculator.update(eval_ner(pred_ner, gold_ner))


    ner_f1 = ner_f1_caculator()


    print(f'\r{dt.now()} Epoch: {epoch_num} Test:{batch} \
            ner F1:{ner_f1:.4f}', end='')
print()
save_test_result(save_dir_path, ner_f1, args)
    
