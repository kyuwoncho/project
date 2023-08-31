from operator import itemgetter
from scipy.sparse import coo_matrix, save_npz
import numpy as np
from collections import defaultdict

import os
import io
import distutils.dir_util
from collections import Counter
import json
import pickle 
from tqdm import tqdm
import parmap

import gc
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix, load_npz
from multiprocessing import Pool
from time import time
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

def pickle_dump(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(fname):
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    return data

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./" + parent)
    with io.open("./" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)
    return json_obj

def most_popular(playlists, col, topk_count):
    '''가장 많이 등장한 곡/태그를 topk_count만큼 추출하는 함수'''
    c = Counter()
    for doc in playlists:
        c.update(doc[col])
    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]

def remove_seen(seen, l):
    '''l 중에서 seen에 이미 있는 아이템은 제거하는 함수'''
    seen = set(seen)
    return [x for x in l if not (x in seen)]

def cos_sim(A, B):
  return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def neighbor_based_cf(playlist_id):
    '''
    전역 변수 : 
    test_item_indices = {플리id1: [곡idx, 곡idx, ...], ...}
    song_playlist_train_matrix_raw : shape (곡 개수, 플리id 개수) 
    mel_mean
    '''
    item_indices = test_item_indices[playlist_id] # 현재 플리id에 해당하는 [곡idx, 곡idx, ...]

    alpha, beta, theta = 0.9, 0.7, 0.99
    Cr = 0.4 + (100 - np.shape(item_indices)[0]) * 0.0055  # np.shape(item_indices)[0] : 플리에 포함된 곡/태그/제목태그 개수
    # Cr : 0.2 ~ 1 사이의 값
    # Cr : 플리에 포함된 곡/태그/제목태그 개수가 100개일 때 0.4, 1개일 때 0.4 + (100-1)*0.0055 = 0.9445
    if Cr < 0.2:
        Cr = 0.2
    elif Cr > 1:
        Cr = 1

    song_playlist_train_matrix = lil_matrix(song_playlist_train_matrix_raw) # shape (곡/태그/제목태그 개수, 플리id 개수)
    song_playlist_train_matrix[:,p_encode[playlist_id]] = 0  # 해당 플리에 해당하는 모든 행을 0으로 설정 >> 현재 플리와 연관된 곡/태그/제목태그들을 제거하는 역할

    weight = song_playlist_train_matrix[item_indices, :].multiply(np.power(1e-1 + I_list, beta - 1)).sum(axis=0) 
    weight = np.array(weight).flatten() 
    weight = np.power(weight,theta) 
    # >> weight: (플리id 개수, ) : 현재 플리와 연관된 곡/태그/제목태그들의 플리별 가중치
    
    value = song_playlist_train_matrix[item_indices, :].multiply(weight) # shape : (플리id에 해당하는 곡/태그/제목태그 개수, 플리 개수) : 현재 플리 value에 가중치 곱
    value = value.dot(song_playlist_train_matrix.transpose()) # shape : (플리id에 해당하는 곡/태그/제목태그 개수, 곡/태그/제목태그 개수)
    I_song_i = np.power(1e-1+I_song[item_indices], -alpha) # shape : (플리id에 해당하는 곡/태그/제목태그 개수, ) 
    value = value.multiply(I_song_i.reshape((-1,1)))  
    value = value.multiply(np.power(1e-1+I_song,alpha-1)) 
    value = csr_matrix(value) # csr_matrix로 변환
    # >> value: (플리id에 해당하는 곡 개수, 곡 개수) 
    
    ## song CF 모델
    predictions = lil_matrix(value) # value를 lil_matrix로 변환
    label = np.zeros(song_playlist_train_matrix.shape[0]) # 곡/태그/제목태그 개수만큼 0으로 채워진 배열
    label[item_indices] = 1  # 해당 플리id 곡/태그/제목태그 위치에 1을 넣음
    # >> label: (곡/태그/제목태그 개수, )
    
    clf = LinearSVC(C=Cr, class_weight={0:1,1:1}, tol=1e-6, dual = True, max_iter=360000) # C: Regularization parameter 
    clf.fit(predictions.transpose(),label)
    predictions = clf.decision_function(predictions.transpose()) # proba
    
    ## mel 유사도 모델
    # 코사인 유사도 계산 
    mel_sim = value.copy()
    # mel_sim = np.zeros(shape=(value.shape)) # mel 유사도를 담을 배열 생성 (0으로 초기화)
    for i in range(len(item_indices)):  # 행
        for j in range(mel_sim.shape[1]):  # 열
            try: 
                mel_sim[i,j] = cos_sim(mel_mean[item_indices[i]], mel_mean[j]) # mel 유사도 계산
            except:
                mel_sim[i,j] = 0
    
    xgb_clf = XGBClassifier(n_estimators=500)
    xgb_clf.fit(mel_sim.transpose(), label)
    # 예측하기, 확률값으로 반환됨
    xgb_predictions = xgb_clf.predict_proba(mel_sim.transpose())[:,1]

    # 두 모델 앙상블
    predictions = predictions + xgb_predictions
    predictions = np.argsort(np.array(predictions).flatten() - min(predictions))[::-1] # 내림차순 정렬된 인덱스 반환

    # 출력 : 곡 400개 인덱스
    return np.array(list(predictions[:400]))  


####### main #######

# preprocessing 불러오기
s_decode = pickle_load('/home/nsun/mountain/data/song_label_decoder_with_mel.pickle')
p_encode = pickle_load('/home/nsun/mountain/data/playlist_label_encoder_with_mel.pickle')
with open('/home/nsun/mountain/data/mel_mean_idx.pkl', 'rb') as f:
    mel_mean = pickle.load(f)

# playlist_song_train_matrix : [플리idx, 곡idx] sparse matrix
print("load train matrix...")
playlist_song_train_matrix = load_npz('/home/nsun/mountain/data/playlist_song_train_matrix_with_mel.npz')  # shape : (플리id 개수, 곡 개수)

# song_playlist_train_matrix_raw : [곡idx, 플리idx] sparse matrix
song_playlist_train_matrix_raw = lil_matrix(playlist_song_train_matrix.transpose()) # shape : (곡 개수, 플리id 개수)

gc.collect()  # 메모리 정리 (불필요하게 쌓인 메모리를 해제)

I_song = np.array(song_playlist_train_matrix_raw.sum(axis=1)).flatten() # shape : (곡 개수, )
I_list = np.array(song_playlist_train_matrix_raw.sum(axis=0)).flatten() # shape : (플리id 개수, )

# # 테스트셋 불러오기
# print("load test data...")
test = load_json('/home/nsun/mountain/data/val_items.json')

# 테스트셋에 items 키가 있는 경우
# test_item_indices = {플리id1: [곡idx, 곡idx, ...], ...}
# test_playlist_id = [플리id1, 플리id2, ...]
test_item_indices = dict()
test_playlist_id = []
for q in test:
    if 'items' in q.keys():
        test_item_indices[q['id']] = q['items']
        test_playlist_id.append(q['id'])


# 예측 시작
print("predictions begin...")
# 멀티프로세싱 - test_playlist_id를 인자로 받아 neighbor_based_cf 함수를 병렬로 실행한 결과를 results에 저장
results = parmap.map(neighbor_based_cf, test_playlist_id, pm_pbar=True, pm_processes=23)

# 예측 결과 딕셔너리 생성
# test_playlist_id = [플리id1, 플리id2, ...]
# prediction_results : {플리id1: {"songs": [곡id1, 곡id2, ..., 곡id400], ...}
prediction_results = {}
for i in tqdm(range(len(results))):
    prediction_results[test_playlist_id[i]] = {"songs": [s_decode[s] for s in results[i][:400]]}


# 예측 결과 저장
# 테스트셋의 플리id가 예측 결과에 있는 경우 -> (예측 결과에 있는 곡/태그 - 테스트셋 플리에 있는 곡/태그) 곡 100개를 answers에 추가
# 테스트셋의 플리id가 예측 결과에 없는 경우 -> (가장 인기있는 곡/태그 - 테스트셋 플리에 있는 곡/태그) 곡 100개를 answers에 추가
print("write results.json...")
answers = []
for q in test:
    if q['id'] in test_playlist_id:
        answers.append({'id': q['id'],
        'songs': remove_seen(q['songs'], prediction_results[q['id']]['songs'])[:100]})
    else:
        answers.append({'id': q['id'],
        'songs': remove_seen(q['songs'], q['songs_mp'])[:100]})
    # 가장 최근에 추가한 플리의 곡 100개가 안되는 경우 -> (answers에 있는 곡/태그 + (가장 인기있는 곡/태그 - 플리에 있는 곡/태그)) 곡 100개로 answers 대체
    if len(answers[len(answers)-1]['songs']) < 10:
        answers[len(answers)-1]['songs'] = (answers[len(answers)-1]['songs'] + remove_seen(q['songs'] + answers[len(answers)-1]['songs'], q['songs_mp']))[:100]

# answers를 json 파일로 저장
write_json(answers, '/home/nsun/mountain/data/results.json')
print('end')
