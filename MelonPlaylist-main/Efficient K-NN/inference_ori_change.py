# 참고 논문: Efficient K-NN for Playlist Continuation (RecSys'18 Challenge)
# 주소: https://eprints.sztaki.hu/9560/1/Kelen_1_30347064_ny.pdf
# 참고 Notebook: 제목으로 태그 맞추기 with Khaiii, Colab
# 주소: https://arena.kakao.com/forum/topics/226
# 참고 Notebook: Melon Playlist Continuation 대회 데이터 전처리 & EDA
# 주소: https://arena.kakao.com/forum/topics/191


import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse

meta = pd.read_json('/data/mountain/song_meta.json', encoding='utf-8')
meta.drop(['album_id', 'album_name', 'artist_id_basket', 'artist_name_basket', 'song_name'], axis=1, inplace=True)

genre = pd.read_json('/data/mountain/genre_gn_all.json', typ='Series', encoding='utf-8')
genre = pd.DataFrame(genre, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})

gnr_big_code = genre[genre['gnr_code'].str[-2:] == '00']
gnr_small_code = genre[genre['gnr_code'].str[-2:] != '00']

gnr_big_code = list(gnr_big_code['gnr_code']) + ['GN9000']
gnr_small_code = list(gnr_small_code['gnr_code'])

r_big_dict = {}
r_small_dict = {}

for i in tqdm(range(len(gnr_big_code))):
    r_big_dict[gnr_big_code[i]] = i

for i in tqdm(range(len(gnr_small_code))):
    r_small_dict[gnr_small_code[i]] = i

big_len = len(gnr_big_code)
small_len = len(gnr_small_code)

train = pd.read_json('/home/q1cho/kakao_arena/거위/data/train_with_tt_token3.json', orient='table', encoding='utf-8')
val = pd.read_json('/home/q1cho/kakao_arena/거위/data/test_with_tt_token3.json', orient='table', encoding='utf-8')

big_dict = {}
small_dict = {}
for i in tqdm(range(meta.shape[0])):
    idx = meta['id'].iloc[i]
    big_dict[idx] = list(map(r_big_dict.get, meta['song_gn_gnr_basket'].iloc[i]))
    small_dict[idx] = list(map(r_small_dict.get, meta['song_gn_dtl_gnr_basket'].iloc[i]))

train_big = []
train_small = []
for i in tqdm(range(train.shape[0])):
    tracks = train['songs'].iloc[i]
    train_big.append(list(set(sum(list(map(big_dict.get, tracks)), []))))
    train_small.append(list(set(sum(list(map(small_dict.get, tracks)), []))))
    
val_big = []
val_small = []
for i in tqdm(range(val.shape[0])):
    tracks = val['songs'].iloc[i]
    val_big.append(list(set(sum(list(map(big_dict.get, tracks)), []))))
    val_small.append(list(set(sum(list(map(small_dict.get, tracks)), []))))

train['big'] = train_big
train['small'] = train_small
val['big'] = val_big
val['small'] = val_small

def combine_tokens(df):
    temp = set(df['title_token']) | set(df['tags_token'])
    return list(temp)

train['tokens'] = train.apply(combine_tokens, axis=1)
val['tokens'] = val.apply(combine_tokens, axis=1)

def convert_to_month(df):
    temp = df['updt_date'].split('-')[:2]
    year = int(temp[0])
    month = int(temp[1])
    year = (year - 2000) * 12
    return year + month

def convert_to_yyyymmdd(df):
    return int(''.join(df['updt_date'].split(' ')[0].split('-')))

train['month'] = train.apply(convert_to_month, axis=1)
val['month'] = val.apply(convert_to_month, axis=1)

train['yyyymmdd'] = train.apply(convert_to_yyyymmdd, axis=1)
val['yyyymmdd'] = val.apply(convert_to_yyyymmdd, axis=1)

train.drop('updt_date', axis=1, inplace=True)
val.drop('updt_date', axis=1, inplace=True)

tracks_len = 707989

all_tags = set([])
all_tokens = set([])
#all_tags_tokens = set([])

for i in tqdm(range(train.shape[0])):
    all_tags = all_tags | set(train['tags'].iloc[i])
    all_tokens = all_tokens | set(train['tokens'].iloc[i])
    
for i in tqdm(range(val.shape[0])):
    all_tags = all_tags | set(val['tags'].iloc[i])
    all_tokens = all_tokens | set(val['tokens'].iloc[i])

all_tags = list(all_tags)
all_tags.sort()
all_tokens = list(all_tokens)
all_tokens.sort()

tags_len = len(all_tags)
tokens_len = len(all_tokens)

tags_dict = {}
tokens_dict = {}

for i in tqdm(range(tags_len)):
    tags_dict[all_tags[i]] = i
    
for i in tqdm(range(tokens_len)):
    tokens_dict[all_tokens[i]] = i

r_tags_dict = {}
r_tokens_dict = {}

for k, v in tqdm(tags_dict.items()):
    r_tags_dict[v] = k
    
for k, v in tqdm(tokens_dict.items()):
    r_tokens_dict[v] = k

train_length_s = []
train_cols_s = []
for i in tqdm(range(train.shape[0])):
    train_cols_s.extend(train['songs'].iloc[i])
    train_cols_s.extend((np.array(list(map(tags_dict.get, train['tags'].iloc[i])))+tracks_len).tolist())
    train_cols_s.extend((np.array(list(map(tokens_dict.get, train['tokens'].iloc[i])))+tracks_len+tags_len).tolist())
    train_cols_s.extend((np.array(train['big'].iloc[i])+tracks_len+tags_len+tokens_len).tolist())
    train_length_s.append(len(train['songs'].iloc[i]) + len(train['tags'].iloc[i]) + len(train['tokens'].iloc[i]) + len(train['big'].iloc[i]))

train_length_t = []
train_cols_t = []
for i in tqdm(range(train.shape[0])):
    train_cols_t.extend(train['songs'].iloc[i])
    train_cols_t.extend((np.array(list(map(tags_dict.get, train['tags'].iloc[i])))+tracks_len).tolist())
    train_cols_t.extend((np.array(list(map(tokens_dict.get, train['tokens'].iloc[i])))+tracks_len+tags_len).tolist())
    train_cols_t.extend((np.array(train['small'].iloc[i])+tracks_len+tags_len+tokens_len).tolist())
    train_length_t.append(len(train['songs'].iloc[i]) + len(train['tags'].iloc[i]) + len(train['tokens'].iloc[i]) + len(train['small'].iloc[i]))

val_length_s = []
val_cols_s = []
for i in tqdm(range(val.shape[0])):
    val_cols_s.extend(val['songs'].iloc[i])
    val_cols_s.extend((np.array(list(map(tags_dict.get, val['tags'].iloc[i])))+tracks_len).tolist())
    val_cols_s.extend((np.array(list(map(tokens_dict.get, val['tokens'].iloc[i])))+tracks_len+tags_len).tolist())
    val_cols_s.extend((np.array(val['big'].iloc[i])+tracks_len+tags_len+tokens_len).tolist())
    val_length_s.append(len(val['songs'].iloc[i]) + len(val['tags'].iloc[i]) + len(val['tokens'].iloc[i]) + len(val['big'].iloc[i]))

val_length_t = []
val_cols_t = []
for i in tqdm(range(val.shape[0])):
    val_cols_t.extend(val['songs'].iloc[i])
    val_cols_t.extend((np.array(list(map(tags_dict.get, val['tags'].iloc[i])))+tracks_len).tolist())
    val_cols_t.extend((np.array(list(map(tokens_dict.get, val['tokens'].iloc[i])))+tracks_len+tags_len).tolist())
    val_cols_t.extend((np.array(val['small'].iloc[i])+tracks_len+tags_len+tokens_len).tolist())
    val_length_t.append(len(val['songs'].iloc[i]) + len(val['tags'].iloc[i]) + len(val['tokens'].iloc[i]) + len(val['small'].iloc[i]))

values = [1 for i in range(sum(train_length_s))]
rows_idx = []
for i in tqdm(range(len(train_length_s)+1)):
    rows_idx.append(sum(train_length_s[:i]))
train_spas_s = sparse.csr_matrix((values, train_cols_s, rows_idx), shape=(train.shape[0], tracks_len+tags_len+tokens_len+big_len))

values = [1 for i in range(sum(train_length_t))]
rows_idx = []
for i in tqdm(range(len(train_length_t)+1)):
    rows_idx.append(sum(train_length_t[:i]))
train_spas_t = sparse.csr_matrix((values, train_cols_t, rows_idx), shape=(train.shape[0], tracks_len+tags_len+tokens_len+small_len))


values = [1 for i in range(sum(val_length_s))]
rows_idx = []
for i in tqdm(range(len(val_length_s)+1)):
    rows_idx.append(sum(val_length_s[:i]))
val_spas_s = sparse.csr_matrix((values, val_cols_s, rows_idx), shape=(val.shape[0], tracks_len+tags_len+tokens_len+big_len))

values = [1 for i in range(sum(val_length_t))]
rows_idx = []
for i in tqdm(range(len(val_length_t)+1)):
    rows_idx.append(sum(val_length_t[:i]))
val_spas_t = sparse.csr_matrix((values, val_cols_t, rows_idx), shape=(val.shape[0], tracks_len+tags_len+tokens_len+small_len))

p_ = 0.4
al_ = 2.1
k_ = 300
n_ = 350

songs_cnt = 100
tags_cnt = 10

train_spas_s_s = train_spas_s[:, :tracks_len]
train_spas_t_t = train_spas_t[:, tracks_len:tracks_len+tags_len+tokens_len]


##########
f_s = train_spas_s.sum(axis=0) - 1
f_s = np.power(f_s, p_) + 1
f_s = np.float_power(f_s, -1)
f_s = np.nan_to_num(f_s)

B2_s = np.sqrt(train_spas_s.sum(axis=1))
fb_s = train_spas_s.multiply(f_s)
##########
f_t = train_spas_t.sum(axis=0) - 1
f_t = np.power(f_t, p_) + 1
f_t = np.float_power(f_t, -1)
f_t = np.nan_to_num(f_t)

B2_t = np.sqrt(train_spas_t.sum(axis=1))
fb_t = train_spas_t.multiply(f_t)
##########
f_s_s = train_spas_s_s.sum(axis=0) - 1
f_s_s = np.power(f_s_s, p_) + 1
f_s_s = np.float_power(f_s_s, -1)
f_s_s = np.nan_to_num(f_s_s)

B2_s_s = np.sqrt(train_spas_s_s.sum(axis=1))
fb_s_s = train_spas_s_s.multiply(f_s_s)
##########
f_t_t = train_spas_t_t.sum(axis=0) - 1
f_t_t = np.power(f_t_t, p_) + 1
f_t_t = np.float_power(f_t_t, -1)
f_t_t = np.nan_to_num(f_t_t)

B2_t_t = np.sqrt(train_spas_t_t.sum(axis=1))
fb_t_t = train_spas_t_t.multiply(f_t_t)
##########


result_id = []
result_songs = []
result_tags = []

for i in tqdm(range(val.shape[0])):
    month_dist = train['month'] - val['month'].iloc[i]
    like_dist = np.sqrt(train['like_cnt']) - np.sqrt(val['like_cnt'].iloc[i])
    dist = abs(month_dist) + abs(like_dist) + 1
    dist = 1 / (dist * n_)
    
    a_s = val_spas_s[i]
    a_t = val_spas_t[i]
    a_s_s = a_s[:, :tracks_len]
    a_t_t = a_t[:, tracks_len:tracks_len+tags_len+tokens_len]
    
    ####################
    A2_s = np.sqrt(a_s.power(2).sum())
    if A2_s > 0:
        AB_s = A2_s * B2_s
        fba_s = fb_s.multiply(a_s)
        fba_s = fba_s.sum(axis=1)
        sim_s = fba_s / AB_s
        sim_s = np.array(sim_s).flatten()
        sim_s_min = sim_s.min()
        diff_s = sim_s.max() - sim_s_min
        if diff_s > 0:
            sim_s = (sim_s - sim_s_min) / diff_s
            sim_s = np.power(sim_s, al_)
            sim_s = np.nan_to_num(sim_s)
            sim_s_arg = np.argsort(sim_s)[::-1][:k_].tolist()
            sim_s = np.sort(sim_s)[::-1][:k_].tolist()
        else:
            sim_s = dist.values
            sim_s_arg = np.argsort(sim_s)[::-1][:k_].tolist()
            sim_s = np.sort(sim_s)[::-1][:k_].tolist()
    else:
        sim_s = dist.values
        sim_s_arg = np.argsort(sim_s)[::-1][:k_].tolist()
        sim_s = np.sort(sim_s)[::-1][:k_].tolist()

    sims_s_k_ = [0 for j in range(train_spas_s.shape[0])]
    for j in range(len(sim_s_arg)):
        sims_s_k_[sim_s_arg[j]] = sim_s[j]
    ####################
    A2_s_s = np.sqrt(a_s_s.power(2).sum())
    if A2_s_s > 0:
        AB_s_s = A2_s_s * B2_s_s
        fba_s_s = fb_s_s.multiply(a_s_s)
        fba_s_s = fba_s_s.sum(axis=1)
        sim_s_s = fba_s_s / AB_s_s
        sim_s_s = np.array(sim_s_s).flatten()
        sim_s_s_min = sim_s_s.min()
        diff_s_s = sim_s_s.max() - sim_s_s_min
        if diff_s_s > 0:
            sim_s_s = (sim_s_s - sim_s_s_min) / diff_s_s
            sim_s_s = np.power(sim_s_s, al_)
            sim_s_s = np.nan_to_num(sim_s_s)
            sim_s_s_arg = np.argsort(sim_s_s)[::-1][:k_].tolist()
            sim_s_s = np.sort(sim_s_s)[::-1][:k_].tolist()
        else:
            sim_s_s = dist.values
            sim_s_s_arg = np.argsort(sim_s_s)[::-1][:k_].tolist()
            sim_s_s = np.sort(sim_s_s)[::-1][:k_].tolist()
    else:
        sim_s_s = dist.values
        sim_s_s_arg = np.argsort(sim_s_s)[::-1][:k_].tolist()
        sim_s_s = np.sort(sim_s_s)[::-1][:k_].tolist()

    sims_s_s_k_ = [0 for j in range(train_spas_s_s.shape[0])]
    for j in range(len(sim_s_s_arg)):
        sims_s_s_k_[sim_s_s_arg[j]] = sim_s_s[j]
    ####################
    A2_t = np.sqrt(a_t.power(2).sum())
    if A2_t > 0:
        AB_t = A2_t * B2_t
        fba_t = fb_t.multiply(a_t)
        fba_t = fba_t.sum(axis=1)
        sim_t = fba_t / AB_t
        sim_t = np.array(sim_t).flatten()
        sim_t_min = sim_t.min()
        diff_t = sim_t.max() - sim_t_min
        if diff_t > 0:
            sim_t = (sim_t - sim_t_min) / diff_t
            sim_t = np.power(sim_t, al_)
            sim_t = np.nan_to_num(sim_t)
            sim_t_arg = np.argsort(sim_t)[::-1][:k_].tolist()
            sim_t = np.sort(sim_t)[::-1][:k_].tolist()
        else:
            sim_t = dist.values
            sim_t_arg = np.argsort(sim_t)[::-1][:k_].tolist()
            sim_t = np.sort(sim_t)[::-1][:k_].tolist()
    else:
        sim_t = dist.values
        sim_t_arg = np.argsort(sim_t)[::-1][:k_].tolist()
        sim_t = np.sort(sim_t)[::-1][:k_].tolist()

    sims_t_k_ = [0 for j in range(train_spas_t.shape[0])]
    for j in range(len(sim_t_arg)):
        sims_t_k_[sim_t_arg[j]] = sim_t[j]
    ####################
    A2_t_t = np.sqrt(a_t_t.power(2).sum())
    if A2_t_t > 0:
        AB_t_t = A2_t_t * B2_t_t
        fba_t_t = fb_t_t.multiply(a_t_t)
        fba_t_t = fba_t_t.sum(axis=1)
        sim_t_t = fba_t_t / AB_t_t
        sim_t_t = np.array(sim_t_t).flatten()
        sim_t_t_min = sim_t_t.min()
        diff_t_t = sim_t_t.max() - sim_t_t_min
        if diff_t_t > 0:
            sim_t_t = (sim_t_t - sim_t_t_min) / diff_t_t
            sim_t_t = np.power(sim_t_t, al_)
            sim_t_t = np.nan_to_num(sim_t_t)
            sim_t_t = sim_t_t + dist.values
            sim_t_t_arg = np.argsort(sim_t_t)[::-1][:k_].tolist()
            sim_t_t = np.sort(sim_t_t)[::-1][:k_].tolist()
        else:
            sim_t_t = dist.values
            sim_t_t_arg = np.argsort(sim_t_t)[::-1][:k_].tolist()
            sim_t_t = np.sort(sim_t_t)[::-1][:k_].tolist()
    else:
        sim_t_t = dist.values
        sim_t_t_arg = np.argsort(sim_t_t)[::-1][:k_].tolist()
        sim_t_t = np.sort(sim_t_t)[::-1][:k_].tolist()

    sims_t_t_k_ = [0 for j in range(train_spas_t_t.shape[0])]
    for j in range(len(sim_t_t_arg)):
        sims_t_t_k_[sim_t_t_arg[j]] = sim_t_t[j]
    ####################
    sims_s_k_ = np.array(sims_s_k_)
    sims_s_s_k_ = np.array(sims_s_s_k_)
    sims_t_k_ = np.array(sims_t_k_)
    sims_t_t_k_ = np.array(sims_t_t_k_)
    ##############################
    sims_k_1 = ((0.52*sims_s_k_) + (0.48*sims_s_s_k_)).tolist()
    ##############################
    rel_1 = train_spas_s[:, :tracks_len].transpose().multiply(sims_k_1).transpose().tocsr()
    rel_t = train_spas_t[:, tracks_len:tracks_len+tags_len].transpose().multiply(sims_t_k_).transpose().tocsr()
    rel_t_t = train_spas_t[:, tracks_len:tracks_len+tags_len].transpose().multiply(sims_t_t_k_).transpose().tocsr()
    
    songs_rel = np.array(rel_1.sum(axis=0).tolist()[0]) / sum(sims_k_1)
    s_rel_t = np.array(rel_t.sum(axis=0).tolist()[0]) / sum(sims_t_k_)
    s_rel_t_t = np.array(rel_t_t.sum(axis=0).tolist()[0]) / sum(sims_t_t_k_)
    tags_rel = (0.65*s_rel_t) + (0.35*s_rel_t_t)
    
    songs_list = songs_rel.argsort()[::-1]
    tags_list = tags_rel.argsort()[::-1]
    
    result_id.append(int(val['id'].iloc[i]))
    
    count = 0
    result = []
    for song in songs_list:
        if song not in val['songs'].iloc[i] and val['yyyymmdd'].iloc[i] >= meta[ meta['id'] == song ]['issue_date'].iloc[0]:
            result.append(int(song))
            count += 1
        if count >= songs_cnt:
            break
    result_songs.append(result)
    
    count = 0
    result = []
    for tag in tags_list:
        tag_ = r_tags_dict[tag]
        if tag_ not in val['tags'].iloc[i]:
            result.append(tag_)
            count += 1
        if count >= tags_cnt:
            break
    result_tags.append(result)

result_df = pd.DataFrame({'id': result_id,
                         'songs': result_songs,
                         'tags': result_tags})

import json

def submission(df):
    final = df[['id', 'songs', 'tags']].to_dict('index')
    final = [i for i in final.values()]
    with open('/data/mountain/goose/results_change_1.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(final, ensure_ascii=False))

submission(result_df)
