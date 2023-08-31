import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from box import Box

import warnings

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)

path = '/data/mountain/'

config = {
    'data_path' : path+'train_bert4rec_gnr_input.csv', # 데이터 경로
    'max_len' : 50,
    'hidden_units' : 50, # Embedding size
    'num_heads' : 1, # Multi-head layer 의 수 (병렬 처리)
    'num_layers': 2, # block의 개수 (encoder layer의 개수)
    'dropout_rate' : 0.5, # dropout 비율
    'lr' : 0.001,
    'batch_size' : 16,
    'num_epochs' : 5,
    'num_workers' : 1,
    'mask_prob' : 0.15, # for cloze task
}

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

config = Box(config)

class MakeSequenceDataSet():
    """
    SequenceData 생성
    - user : 플레이리스트
    - item : 노래
    - gnr : 장르
    """
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(self.config.data_path)
        
        self.item_encoder, self.item_decoder = self.generate_encoder_decoder('song_id')
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder('playlist_id')
        self.gnr_encoder, self.gnr_decoder = self.generate_encoder_decoder('song_gn_gnr')
        self.num_item, self.num_user,self.num_gnr = len(self.item_encoder), len(self.user_encoder),len(self.gnr_encoder)

        self.df['song_idx'] = self.df['song_id'].apply(lambda x : self.item_encoder[x] + 1)
        self.df['playlist_idx'] = self.df['playlist_id'].apply(lambda x : self.user_encoder[x])
        self.df['gnr_idx'] = self.df['song_gn_gnr'].apply(lambda x : self.gnr_encoder[x] + 1)
        self.df = self.df.sort_values(['playlist_idx', 'order']) # 순서에 따라 정렬
        self.user_train, self.user_valid ,self.gnr_train,self.gnr_valid= self.generate_sequence_data()

    def generate_encoder_decoder(self, col : str) -> dict:
        """
        encoder, decoder 생성

        Args:
            col (str): 생성할 columns 명
        Returns:
            dict: 생성된 user encoder, decoder
        """

        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder
    
    def generate_sequence_data(self) -> dict:
        """
        sequence_data 생성
        - user : 플레이리스트
        - item : 노래
        - gnr : 장르

        Returns:
            dict: train user sequence / valid user sequence
        """
        # 플리별로 노래 시퀀스 리스트업
        users = defaultdict(list)  # 딕셔너리 value를 list로 초기화
        user_train = {}
        user_valid = {}
        
        group_df = self.df.groupby('playlist_idx')
        for user, item in group_df:
            users[user].extend(item['song_idx'].tolist()) 
            
        
        # 유저별로 마지막 아이템은 valid, 이전 아이템까지는 train으로 분리
        for user in users:
            user_train[user] = users[user][:-1]
            user_valid[user] = [users[user][-1]] # 마지막 아이템을 예측

        # 플리별로 노래 시퀀스 리스트업
        gnrs = defaultdict(list)  # 딕셔너리 value를 list로 초기화
        gnr_train = {}
        gnr_valid = {}
        
        group_df = self.df.groupby('playlist_idx')
        for user, item in group_df:
            gnrs[user].extend(item['gnr_idx'].tolist()) 
            
        
        # 유저별로 마지막 아이템은 valid, 이전 아이템까지는 train으로 분리
        for user in users:
            gnr_train[user] = gnrs[user][:-1]
            gnr_valid[user] = [gnrs[user][-1]] # 마지막 아이템을 예측

        return user_train, user_valid, gnr_train,gnr_valid
    
    def get_train_valid_data(self):
        return self.user_train, self.user_valid,self.gnr_train,self.gnr_valid
    
make_sequence_dataset = MakeSequenceDataSet(config = config)
user_train, user_valid, gnr_train, gnr_valid = make_sequence_dataset.get_train_valid_data()

gnr_dict = {}

for u in user_train:
  for i in range(len(user_train[u])):
    if user_train[u][i] not in gnr_dict:
      gnr_dict[user_train[u][i]] = gnr_train[u][i]

gnr_dict[0] = 0

class BERTRecDataSet(Dataset):
    def __init__(self, user_train,genre_train,max_len, num_user, num_item,num_genre,mask_prob):
        self.user_train = user_train
        self.genre_train = genre_train
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item
        self.num_genre = num_genre
        self.mask_prob = mask_prob
        self._all_items = set([i for i in range(1, self.num_item + 1)])

    def __len__(self):
        # 데이터셋의 길이 (총 샘플의 수) 즉, len(dataset)을 했을 때 데이터셋의 크기를 리턴할 len
        # 총 user(플레이리스트)의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): 
        # 데이터셋에서 특정 1개의 샘플을 가져오는 함수 즉, dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱을 위한 get_item
        user_seq = self.user_train[user]
        genre_seq = self.genre_train[user]
        tokens = []
        labels = []
        for s in user_seq[-self.max_len:]: # 최근 n개 아이템만 
            prob = np.random.random() 
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    # noise
                    tokens.extend(self.random_neg_sampling(rated_item = user_seq, num_item_sample = 1))  # item random sampling 1개
                else:
                    # original
                    tokens.append(s)
                labels.append(s) # 학습에 사용 O
            else:
                tokens.append(s)
                labels.append(0) # 학습에 사용 X

        # padding
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        genres = []
        for t in tokens:
          try:
            genres.append(gnr_dict[t])
          except:
            genres.append(self.num_genre+1)

        return torch.LongTensor(tokens), torch.LongTensor(genres),torch.LongTensor(labels)

    def random_neg_sampling(self, rated_item : list, num_item_sample : int):
        '''
        플레이리스트에 없는 노래를 랜덤으로 num_item_sample 개수만큼 샘플링 
        '''
        nge_samples = random.sample(list(self._all_items - set(rated_item)), num_item_sample)
        return nge_samples

bert4rec_dataset = BERTRecDataSet(
    user_train = user_train, 
    genre_train = gnr_train,
    max_len = config.max_len, 
    num_user = make_sequence_dataset.num_user, 
    num_item = make_sequence_dataset.num_item,
    num_genre = make_sequence_dataset.num_gnr,
    mask_prob = config.mask_prob,
    )

data_loader = DataLoader(
    bert4rec_dataset, 
    batch_size = config.batch_size, 
    shuffle = True, 
    pin_memory = True,
    num_workers = config.num_workers,
    )

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask):
        """
        Q, K, V : (batch_size, num_heads, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        """
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units) # (batch_size, num_heads, max_len, max_len)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        output = torch.matmul(attn_dist, V)  # (batch_size, num_heads, max_len, hidden_units) / # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units
        
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_O = nn.Linear(hidden_units * num_heads, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        """
        enc : (batch_size, max_len, hidden_units)
        mask : (batch_size, 1, max_len, max_len)
        
        """
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)

        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        # view() : tensor의 shape을 변경함 (batch_size, max_len, hidden_units) -> (batch_size, max_len, num_heads, hidden_units)
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) # (batch_size, max_len, num_heads, hidden_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) # (batch_size, max_len, num_heads, hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) # (batch_size, max_len, num_heads, hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2) # (batch_size, num_heads, max_len, hidden_units)
        output, attn_dist = self.attention(Q, K, V, mask) # output : (batch_size, num_heads, max_len, hidden_units) / attn_dist : (batch_size, num_heads, max_len, max_len)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합침
        # continuous() : 가변적 메모리 할당
        output = output.transpose(1, 2).contiguous() # (batch_size, max_len, num_heads, hidden_units) / contiguous() : 가변적 메모리 할당
        output = output.view(batch_size, seqlen, -1) # (batch_size, max_len, hidden_units * num_heads)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual) # (batch_size, max_len, hidden_units)
        return output, attn_dist
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()

        self.W_1 = nn.Linear(hidden_units, hidden_units)
        self.W_2 = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, x):
        residual = x
        # Feed-Forward Network
        output = self.W_2(F.relu(self.dropout(self.W_1(x))))
        # Add & Norm
        output = self.layerNorm(self.dropout(output) + residual)
        return output


class BERT4RecBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist

class BERT4Rec(nn.Module):
    def __init__(self, num_user, num_item,num_genre, hidden_units, num_heads, num_layers, max_len, dropout_rate, device):
        super(BERT4Rec, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_genre = num_genre
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers 
        self.device = device
        self.genre_emb = nn.Embedding(num_genre+2,hidden_units)
        self.item_emb = nn.Embedding(num_item + 2, hidden_units, padding_idx=0) # padding : 0 / item : 1 ~ num_item + 1 /  mask : num_item + 2
        self.pos_emb = nn.Embedding(max_len, hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)
        
        self.blocks = nn.ModuleList([BERT4RecBlock(num_heads, hidden_units, dropout_rate) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_units, num_item + 1)
    
    def forward(self, log_seqs,genre_seqs):
        """
        log_seqs : (batch_size, max_len)

        ex)
        log_seqs = [
                [1, 2, 3, 4, 5],
                [0, 0, 0, 1, 2],
                [0, 0, 1, 2, 3]
        ]
        
        """
        genre = self.genre_emb(torch.LongTensor(genre_seqs).to(self.device))

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device)) # (batch_size, max_len, hidden_units)
        
        seqs = genre+seqs
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]) # log_seqs의 max_len을 (batch_size, max_len) 크기로 복사, 각 원소는 position 순서
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device)) # (batch_size, max_len, hidden_units)
        seqs = self.emb_layernorm(self.dropout(seqs)) # LayerNorm

        # Mask for zero pad
        # BoolTensor(log_seqs > 0) : log_seqs의 각 원소가 0보다 크면 True, 아니면 False
        # unsqueeze(1) : (batch_size, max_len) -> (batch_size, 1, max_len)
        # repeat(a,b,c) : 인덱스 a번째 위치에 b를 c번 반복 
        # repeat(1, log_seqs.shape[1], 1) : (batch_size, 1, max_len) -> (batch_size, max_len, max_len)
        mask_pad = torch.BoolTensor(log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device) # (batch_size, 1, max_len, max_len)
        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask_pad)
        out = self.out(seqs) # (batch_size, max_len, num_item + 1)
        return out

def train(model, criterion, optimizer, data_loader):
    model.train()
    loss_val = 0
    for seq, genre,labels in data_loader:
        logits = model(seq,genre) # (batch_size, max_len, num_item + 1)
    
        logits = logits.view(-1, logits.size(-1)) # (batch_size * max_len, num_item + 1)
        labels = labels.view(-1).to(device) # (batch_size * max_len)

        optimizer.zero_grad()
        loss = torch.sqrt(criterion(logits, labels))

        loss_val += loss.item()

        loss.backward()
        optimizer.step()
    
    loss_val /= len(data_loader)

    return loss_val

def evaluate(model, user_train, user_valid, gnr_train,gnr_valid,max_len, bert4rec_dataset, make_sequence_dataset):
    model.eval()

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    num_item_sample = 100

    users = [user for user in range(make_sequence_dataset.num_user)]

    for user in users:
        seq = (user_train[user] + [make_sequence_dataset.num_item + 1])[-max_len:]
        rated = user_train[user] + user_valid[user]
        gnr = (gnr_train[user] + [make_sequence_dataset.num_gnr + 1])[-max_len:]
        # negative sample 100개 샘플링
        items = user_valid[user] + bert4rec_dataset.random_neg_sampling(rated_item = rated, num_item_sample = num_item_sample)

        
        with torch.no_grad():
            predictions = -model(torch.LongTensor(seq).unsqueeze(0),torch.LongTensor(gnr).unsqueeze(0)) # predictions : seq 다음에 나올 아이템들의 확률
            predictions = predictions[0][-1][items] # sampling
            rank = predictions.argsort().argsort()[0].item()

        if rank < 10: #Top10
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    NDCG /= len(users)
    HIT /= len(users)

    return NDCG, HIT

model = BERT4Rec(
    num_user = make_sequence_dataset.num_user, 
    num_item = make_sequence_dataset.num_item, 
    num_genre = make_sequence_dataset.num_gnr,
    hidden_units = config.hidden_units, 
    num_heads = config.num_heads, 
    num_layers = config.num_layers, 
    max_len = config.max_len, 
    dropout_rate = config.dropout_rate, 
    device = device,
    ).to(device)

# model = nn.DataParallel(model, device_ids = [2,3])   # 2개의 GPU를 이용할 경우, 대신 이 코드 쓰면 바로위 model에서 to(device)는 빼야함!
# model.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
#criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)        

loss_list = []
ndcg_list = []
hit_list = []
for epoch in range(1, config.num_epochs + 1):
    tbar = tqdm(range(1))
    for _ in tbar:
        train_loss = train(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            data_loader = data_loader)
        
        ndcg, hit = evaluate(
            model = model, 
            user_train = user_train, 
            user_valid = user_valid,
            gnr_train = gnr_train,
            gnr_valid = gnr_valid,
            max_len = config.max_len,
            bert4rec_dataset = bert4rec_dataset, 
            make_sequence_dataset = make_sequence_dataset,
            )

        loss_list.append(train_loss)
        ndcg_list.append(ndcg)
        hit_list.append(hit)

        tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')

# 모델 저장
torch.save(model.state_dict(), '/home/sjkim/letsmountain/model/bert4rec_gnr_melon.pt') 
