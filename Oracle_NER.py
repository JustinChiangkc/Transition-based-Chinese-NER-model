import codecs
import sys
import argparse
import json
import os



parser = argparse.ArgumentParser(description='Give Me Conll data.')
parser.add_argument('-f', type=str, help='conll file')
parser.add_argument('-o', type=str, help='output folder')
args = parser.parse_args()

def read_corpus_ner(lines, word_count):
	#Generate feature & action list
    # reference: stack_NER/model/utils.py
    features = list()
    actions = list()
    labels = list()
    tmp_fl = list()#word
    tmp_ll = list()#entity
    tmp_al = list()#actionlist
    count_ner = 0
    ner_label = ""
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])################word
            if line[0] in word_count:
                word_count[line[0]] += 1
            else:
                word_count[line[0]] = 1
            tmp_ll.append(line[-1])################entity
            if len(line[-1].split('-')) > 1:
                if line[-1].split('-')[0] == "B" and not ner_label == "":####ner_label???
                    tmp_al.append(ner_label)
                    count_ner += 1
                ner_label = "REDUCE-"+line[-1].split('-')[1].split('.')[0]
                tmp_al.append("SHIFT")
            else:
                if not ner_label == "":
                    tmp_al.append(ner_label)
                    count_ner += 1
                    ner_label = ""
                tmp_al.append("OUT")

        elif len(tmp_fl) > 0:
            if not ner_label =="":
                tmp_al.append(ner_label)
                count_ner += 1
                ner_label = ""
            assert len(tmp_ll) == len(tmp_fl)
            assert len(tmp_al) == len(tmp_fl)+count_ner
            features.append(tmp_fl)
            labels.append(tmp_ll)
            actions.append(tmp_al)
            count_ner = 0
            tmp_al = list()
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        assert len(tmp_ll) == len(tmp_fl)
        assert len(tmp_al) == len(tmp_fl)+count_ner
        features.append(tmp_fl)
        labels.append(tmp_ll)
        actions.append(tmp_al)

    return features, labels, actions, word_count

def conll2parser(dev_features, dev_actions):
	#Trasfer to parsing friendly
    #Reference: https://github.com/clab/stack-lstm-ner/tree/master/ner-system
    all_train_data = []
    for nums in range(len(dev_features)):
    
        sentence, actions = dev_features[nums], dev_actions[nums]
        stack_state, buffer_state = [], [{'word':x} for x in sentence]
        a_train_data = {"seg_sentence":"".join(sentence), "configurations":[], "gold_ner":[]}

        for action in actions:    
            #print("stack_state:",stack_state,"buffer_state:",buffer_state,"Action:",action )
            a_train_data["configurations"].append({"stack":stack_state.copy(),"buffer":buffer_state.copy(),"action":action})
            if action[0] == 'O':
                assert len(stack_state) == 0
                buffer_state = buffer_state[1:]
            elif action[0] == 'S':
                stack_state.append(buffer_state[0])
                buffer_state = buffer_state[1:]
            elif action[0] == 'R':
                a_entity = "".join([w['word'] for w in stack_state])# produce entity word str
                a_train_data["gold_ner"].append([a_entity,action.split('-')[-1].split('.')[0]])
                stack_state = []
        assert len(stack_state) == 0 and len(buffer_state) == 0#Make sure S & B are both empty
        all_train_data.append(a_train_data)
    return all_train_data

with codecs.open(args.f, 'r', 'utf-8') as f:
    lines = f.readlines()
    word_count = dict()
    dev_features, dev_labels, dev_actions, word_count = read_corpus_ner(lines, word_count)
    datas = conll2parser(dev_features, dev_actions)
    
for i, data in enumerate(datas):
    directory = str(args.o)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "{}/{}.json".format(directory,i)
    with open(filename,'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
