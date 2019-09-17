from function import turn_feature_in_text, feature2id
import itertools
import torch
from data_utils import get_id_map, get_pos2id, get_act2id

lexical2id = get_id_map()
#pos2id = get_pos2id()
act2id = get_act2id()

id2act = {v: k for k, v in act2id.items()}
SH_mask = [0 if k.startswith('SH') else 1 for k in act2id]
SH_mask = torch.FloatTensor(SH_mask).cuda()
# AP_mask = [0 if k.startswith('AP') else 1 for k in act2id]
# AP_mask = torch.FloatTensor(AP_mask).cuda()
R_mask = [0 if k.startswith('R') else 1 for k in act2id]
R_mask = torch.FloatTensor(R_mask).cuda()
O_mask = [0 if k.startswith('O') else 1 for k in act2id]
O_mask = torch.FloatTensor(O_mask).cuda()

softmax = torch.nn.Softmax(dim=-1)

##def transition(model, input_is, seg=True, pos=True, dep=True, args=None, debug=False):
def transition_forIII(model, input_is, args=None, debug=False):
    #input_is => str
    model = model.eval()
    ner_result = []
    stack_state, buffer_state = [], [{'word':x} for x in input_is[0]]
    terminate = False
    entity_s = 0
    entity_e = 0
    
    while terminate == False:
        if debug == True:
            print('stack', stack_state)
            print('buffer', buffer_state)
        configuration = turn_feature_in_text(stack_state, buffer_state)#, dep_parsed_graph)
        if debug == True:
            print('configuration:', configuration)
        offsets = []
        wfcs = []
        wfw, wfc, char_mask, lenq0 = feature2id(configuration, lexical2id)
        if debug == True:
            print('wfw:', wfw)
            print('wfc:', wfc)
        for w in wfc:
            wfcs += w
            offsets.append(len(w))
        offsets = (list(itertools.accumulate(offsets, lambda x, y : x+y)))
        offsets.insert(0, 0)
        offsets = offsets[:-1]

        wfws = torch.LongTensor(wfw).unsqueeze(0).cuda()
        wfcs = torch.LongTensor(wfcs).cuda()
        char_masks = torch.LongTensor(char_mask).unsqueeze(0).cuda()
        offsets = torch.LongTensor(offsets).cuda()
        lenq0s = torch.LongTensor([lenq0]).cuda()
        pred_action = model(wfws, wfcs, char_masks, offsets, lenq0s).cuda()
        pred_action = softmax(pred_action)
        if debug == True:
            print('pred_action before all those masks', pred_action)

        if len(stack_state) == 0:
            # cannot do R
            pred_action = pred_action * R_mask
            if debug == True:
                print('pred_action after len(stack_state) == 0, R_mask', pred_action)
        if len(buffer_state) == 0:
            #cannot do SH,O
            pred_action = pred_action * SH_mask
            pred_action = pred_action * O_mask
            if debug == True:
                print('pred_action after len(buffer_state)==0, SH_mask & O_mask', pred_action)

                pred_action.cpu()
        if debug == True:
            print('pred_action in the end', pred_action)
        action = id2act[torch.argmax(pred_action.cpu()).item()]

        pro = pred_action.detach().cpu().numpy()[0,torch.argmax(pred_action.cpu()).item()]


        if debug == True:
            print('action', action)
        if len(buffer_state) == 0 and len(stack_state) == 0 :
            terminate = True
            break


        if action.startswith('O'):
            buffer_state = buffer_state[1:]
            entity_s += 1
            entity_e += 1

        elif action.startswith('SH'):
            stack_state.append(buffer_state[0])
            buffer_state = buffer_state[1:]
            entity_e += 1

        elif action.startswith('RE'):
            a_entity = "".join([w['word'] for w in stack_state])# produce entity word str
            ner_result.append([[entity_s,entity_e-1],a_entity,action.split('-')[-1].split('.')[0], pro])
            stack_state = []
            entity_s = entity_e 
    return ner_result


