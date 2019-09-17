import torch

def feature2id(feature_tuples, lexical2id):
    '''
     turn features in text (e.g. word, char, pos) into id
     for word_features, this function will produce three list, word_id, a char_id and a char_mask, 
     - word id: each word to its id, if unknown in word then is UNK and set corresponding char_mask to 1, otherwise char_mask is 0
     - char id: each character in a word to its id, therefore it's a list of lists.
     - char_mask: indicate it use word or char

    and for pos_features and char_features, it just produce id in each.

    '''
    word_features, lenq0 = feature_tuples
    word_feature_in_wordid = []
    word_feature_in_charid = []
    char_mask = []
    #print('word_features', word_features)
    for text in word_features:
        #print('text is ', text)
        if text == '':
            word_feature_in_wordid.append(lexical2id['PAD'])
            char_mask.append(0)
            word_feature_in_charid.append([lexical2id[x]  if x in lexical2id else lexical2id['UNK'] for x in text])
        elif text not in lexical2id:
            word_feature_in_wordid.append(lexical2id['UNK'])
            char_mask.append(1)
            word_feature_in_charid.append([lexical2id[x]  if x in lexical2id else lexical2id['UNK'] for x in text])
        else:
            word_feature_in_wordid.append(lexical2id[text])
            char_mask.append(0)
            word_feature_in_charid.append([lexical2id[x]  if x in lexical2id else lexical2id['UNK'] for x in text])

    
    return word_feature_in_wordid, word_feature_in_charid, char_mask, lenq0



def find_child(index_word_pos, direction, dep_graph, index):
    if index_word_pos == '':
        # the case that trying to find child of child, but even don't have first child
        return {'index': '', 'word': '', 'pos': '', 'arc_label': ''}
    current_index = index_word_pos['index']
    if current_index == '':
        # the case that trying to find child of child, but even don't have first child
        return {'index': '', 'word': '', 'pos': '', 'arc_label': ''}
    children = []
    dep_graph = sorted(dep_graph, key=lambda x: x[0][0], reverse=True)
    #print('dep_graph is', dep_graph)
    if direction == 'left':
        # x[0] is the child of that dep tuple and x[0][0] is that index

        left_dep = [x for x in dep_graph if x[0][0] < current_index]
        for dep_tuple in left_dep[::-1]:
            if dep_tuple[0][3] == current_index:
                children.append(dep_tuple)
    elif direction == 'right':
        right_dep = [x for x in dep_graph if x[0][0] > current_index]
        for dep_tuple in right_dep:
            if dep_tuple[0][3] == current_index:
                children.append(dep_tuple)
    else:
        return ''
    if len(children) > index:

        return {'index': children[index][0][0], 'word': children[index][0][1], 
        'pos': children[index][0][2], 'arc_label': children[index][0][4]}
    else:
        return {'index': '', 'word': '', 'pos': '', 'arc_label': ''}

def turn_feature_in_text(stack, buffer):
    '''
        state --> feature in text
        ref: Neural Joint Model for Transition-based Chinese 
             Syntactic Analysis
    '''

    s0w = stack[-1]['word'] if len(stack) >= 1 else ''
    s1w = stack[-2]['word'] if len(stack) >= 2 else ''
    s2w = stack[-3]['word'] if len(stack) >= 3 else ''

    # buffer characters
    bcs = ''.join([x['word'] for x in buffer])
    b0c = bcs[0] if len(bcs) >= 1 else ''
    b1c = bcs[1] if len(bcs) >= 2 else ''
    b2c = bcs[2] if len(bcs) >= 3 else ''
    b3c = bcs[3] if len(bcs) >= 4 else ''
    
    # previously shifted words and tags
    q0w = stack[-1]['word'] if len(stack) >= 1 else ''
    q1w = stack[-2]['word'] if len(stack) >= 2 else ''
    # q0p = stack[-1]['pos'] if len(stack) >= 1 else ''
    # q1p = stack[-2]['pos'] if len(stack) >= 2 else ''

    # character of q0:
    q0e = stack[-1]['word'][-1] if len(stack) >= 1 else ''

    # part of q0 word
    q0f1 = q0e[0] if len(q0e) >= 1 else ''
    q0f2 = q0e[0] + q0e[1] if len(q0e) >= 2 else ''
    q0f3 = q0e[0] + q0e[1] + q0e[2] if len(q0e) >= 3 else ''

    #Strings across q0 and buf
    q0 = stack[-1]['word'] if len(stack) >= 1 else ''
    q0b1 = q0 + bcs[0] if len(bcs) >= 1 else ''
    q0b2 = q0 + bcs[0] + bcs[1] if len(bcs) >= 2 else ''
    q0b3 = q0 + bcs[0] + bcs[1] + bcs[2] if len(bcs) >= 3 else ''
    
    # string of buffer characters
    buffer_chars = ''.join([x['word'] for x in buffer])
    b0_2 = ''.join(buffer_chars[0:3])
    b0_3 = ''.join(buffer_chars[0:4])
    b0_4 = ''.join(buffer_chars[0:5])
    b1_3 = ''.join(buffer_chars[1:4])
    b1_4 = ''.join(buffer_chars[1:5])
    b1_5 = ''.join(buffer_chars[1:6])
    b2_4 = ''.join(buffer_chars[2:5])
    b2_5 = ''.join(buffer_chars[2:6])
    b2_6 = ''.join(buffer_chars[2:7])
    b3_5 = ''.join(buffer_chars[3:6])
    b3_6 = ''.join(buffer_chars[3:7])
    b4_6 = ''.join(buffer_chars[4:7])

    lenq0 = len(q0w)
    if lenq0 > 21:
        lenq0 = 21

    return (s0w, s1w, s2w, b0c, b1c, b2c, b3c, q0w, q1w, q0e, q0f1, q0f2, q0f3, q0b1, q0b2, q0b3, b0_2, b0_3, b0_4, b1_3, b1_4, b1_5, b2_4, b2_5, b2_6, b3_5, b3_6, b4_6),\
        lenq0
        