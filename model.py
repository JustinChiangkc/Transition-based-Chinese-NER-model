import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    '''
        The question is the hidden_size of mlp, and how do we combine 
    different dimension features?
    '''
    ##def __init__(self, embedding, hidden, pos_vocab_size, output_class, dropout_rate):
    def __init__(self, embedding, hidden, output_class, dropout_rate):
        super(FFN, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))
        self.word_embedding.freeze = True
        self.char_embedding = nn.EmbeddingBag(embedding.shape[0], embedding.shape[1], mode='mean')
        self.char_embedding.weight = nn.Parameter(torch.from_numpy(embedding).float())
        self.char_embedding.freeze = True
        ##self.pos_embedding = nn.Embedding(pos_vocab_size, 20)
        ##nn.init.xavier_uniform_(self.pos_embedding.weight)
        self.lenq0_embedding = nn.Embedding(22, 20)
        nn.init.xavier_uniform_(self.lenq0_embedding.weight)
        self.mlp1 = nn.Linear(5620, hidden)
        nn.init.xavier_uniform_(self.mlp1.weight)
          
        self.mlp2 = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.mlp2.weight)
          
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden, output_class)
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.dropout = nn.Dropout(p=dropout_rate)        

    ##def forward(self, wfws, wfcs, char_masks, offsets, pfps, lenq0s):
    def forward(self, wfws, wfcs, char_masks, offsets, lenq0s):  
        word_embedding = self.word_embedding(wfws)
        batch, feature_num = wfws.shape[0], wfws.shape[1]
        char_embedding = self.char_embedding(wfcs, offsets)
        char_embedding = char_embedding.view(batch, feature_num, -1)
        char_masks = char_masks.unsqueeze(-1).float()

        word_features = word_embedding * (1-char_masks) + char_embedding * char_masks

        #pos_features = self.pos_embedding(pfps)

        lenq0s = self.lenq0_embedding(lenq0s)

        word_features = word_features.view(word_features.shape[0], -1)
        #pos_features = pos_features.view(pos_features.shape[0], -1)
        features = torch.cat([word_features, lenq0s], dim=-1)
        features = self.dropout(features)
        
        h1 = self.mlp1(features)
        h1 = self.activation(h1)
        h1 = self.dropout(h1)

        h2 = self.mlp2(h1)
        h2 = self.activation(h2)
        h2 = self.dropout(h2)

        output = self.output_layer(h2)

        return output

