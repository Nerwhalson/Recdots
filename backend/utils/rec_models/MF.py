import numpy as np
import pandas as pd
import os, time, math, datetime

REC_PATH = os.path.join(os.getcwd(), "utils", "rec_models", "save")
from .. import models


# ����˵����
#     R:�û�-��Ʒ��Ӧ�Ĺ��־��� m*n
#     P:�û����Ӿ��� m*k
#     Q:��Ʒ���Ӿ��� k*n
#     K:��������ά�� 
#     steps:����������


# ������R�ֽ��P,Q
def matrix_factorization(R, P, Q, K, steps, alpha=0.05, Lambda=0.002):
    # ��ʱ��
    sum_st = 0
    # ǰһ�ε���ʧ��С
    e_old = 0
    # ��������ı�ʶ
    flag = 1
    # �ݶ��½���������1����������������
    for step in range(steps):
        # ÿ�ε�����ʼ��ʱ��
        st = time.time()
        cnt = 0
        e_new = 0
        for u in range(1, len(R)):
            for i in range(1, len(R[u])):
                if R[u][i] > 0:
                    eui = R[u][i] - np.dot(P[u, :], Q[:, i])
                    for k in range(K):
                        temp = P[u][k]
                        P[u][k] = P[u][k] + alpha * eui * Q[k][i] - Lambda * P[u][k]
                        Q[k][i] = Q[k][i] + alpha * eui * temp - Lambda * Q[k][i]
        for u in range(1, len(R)):
            for i in range(1, len(R[u])):
                if R[u][i] > 0:
                    cnt += 1
                    e_new = e_new + pow(R[u][i] - np.dot(P[u, :], Q[:, i]), 2)
        e_new = e_new / cnt
        et = time.time()
        sum_st = sum_st + (et - st)
        # ��һ�ε�����ִ��ǰ����ʧ֮��
        if step == 0:
            e_old = e_new
            continue
        # �ݶ��½���������2��loss��С������
        if e_new < 1e-3:
            flag = 2
            break
        # �ݶ��½���������3��ǰ��loss֮���С������
        if (e_old - e_new) < 1e-10:
            flag = 3
            break
        else:
            e_old = e_new
    print(f'--------Summary---------\nThe type of jump out:{flag}\nTotal steps:{step+1}\nTotal time:{sum_st}\n'
          f'Average time:{sum_st / (step+1)}\nThe e is :{e_new}')
    return P, Q


# �ָ����ݼ���ѵ���������Լ�
def split_data():
    # ��ȡԭʼ����
    data = pd.DataFrame(models.Behavior.objects.values())
    data = data[data['behave_type'] == 'Score']
    rating = data[['user_id', 'item_id', 'score', 'update_time']]
    rating['update_time'].apply(lambda x: x.to_pydatetime().timestamp())
    rating = rating.rename(columns={'user_id': 'user', 'item_id': 'item', 'score': 'score', 'update_time': 'time'})
    # ����ʱ��˳������
    rating.sort_values(by=['time'], axis=0, inplace=True)
    # ����ʱ��˳��ֵ8:2��ȷ���߽���
    boundary1 = rating['time'].quantile(0.8)
    # boundary2 = rating['time'].quantile(0.8)

    # ��ʱ��ֽ���з����ݣ�����ѵ����
    train = rating[rating['time'] < boundary1]
    # ѵ�������û���ʱ��˳������
    train.sort_values(by=['user', 'time'], axis=0, inplace=True)

    # ��ʱ��ֽ���з����ݣ����ɲ��Լ�
    test = rating[rating['time'] >= boundary1]
    # ��֤�����û���ʱ��˳������
    test.sort_values(by=['user', 'time'], axis=0, inplace=True)
    
    data = pd.concat([train, test])

    # ��ѵ���������Լ�д���ļ���
    train.to_csv(os.path.join(REC_PATH, "Train.txt"), sep=',', index=False, header=None)
    test.to_csv(os.path.join(REC_PATH, "Test.txt"), sep=',', index=False, header=None)

    return data, train, test


# ��ȡ��������
def getData(data):
    # ���û�����
    all_user = np.unique(data['user'])
    # ����Ŀ����
    all_item = np.unique(data['item'])
    return all_user, all_item


# �����û�-��Ʒ���󲢱��浽�����ļ���
def getUserItem(train_data, all_user, all_item):
    train_data.sort_values(by=['user', 'item'], axis=0, inplace=True)
    # �û�-��Ŀ���־�������
    num_user = np.max(all_user)+1
    # �û�-��Ŀ���־�������
    num_item = np.max(all_item)+1
    # �û�-��Ŀ���־����ʼ��
    rating_mat = np.zeros([num_user, num_item], dtype=int)
    # �û�-��Ŀ���־���ֵ
    for i in range(len(train_data)):
        user = train_data.iloc[i]['user']
        item = train_data.iloc[i]['item']
        score = train_data.iloc[i]['score']
        rating_mat[user][item] = score
    # �����û�-��Ŀ���־����ļ�
    np.savetxt(os.path.join(REC_PATH, "rating.txt"), rating_mat, fmt='%d', delimiter=',', newline='\n')
    print(f'generate rating matrix complete!')

    return rating_mat


# ����topk�Ƽ��б�
def topK(dic, k):
    keys = []
    values = []
    for i in range(k):
        key, value = max(dic.items(), key=lambda x: x[1])
        keys.append(key)
        values.append(value)
        dic.pop(key)
    return keys, values


#	ѵ��
def train(rating, K, steps):
    R = rating
    M = len(R)
    N = len(R[0])
    # �û������ʼ��
    P = np.random.normal(loc=0, scale=0.01, size=(M, K))
    # ��Ŀ�����ʼ��
    Q = np.random.normal(loc=0, scale=0.01, size=(K, N))
    P, Q = matrix_factorization(R, P, Q, K, steps)
    # ��P��Q���浽�ļ�
    np.savetxt(os.path.join(REC_PATH, "MF_userMatrix.txt"), P, fmt="%.6f", delimiter=',', newline='\n')
    np.savetxt(os.path.join(REC_PATH, "MF_itemMatrix.txt"), Q, fmt="%.6f", delimiter=',', newline='\n')
    print("train complete!")

    return P, Q

# ����
def test(trainData, testData, all_item, k):
    # ��ȡ�û�����
    P = np.loadtxt(os.path.join(REC_PATH, "MF_userMatrix.txt"), delimiter=',', dtype=float)
    # ��ȡ��Ŀ����
    Q = np.loadtxt(os.path.join(REC_PATH, "MF_itemMatrix.txt"), delimiter=',', dtype=float)
    # ���Լ��е��û�����
    testUser = np.unique(testData['user'])
    # ���Լ��ĳ���
    test_lenght = len(testData)

    Hits = 0
    MRR = 0
    NDCG= 0
    # ��ʼʱ��
    st = time.time()
    for user_i in testUser:
        # ���Լ���i���û���ѵ�����ѷ��ʵ���Ŀ
        visited_list = list(trainData[trainData['user'] == user_i]['item'])
        # û��ѵ�����ݣ�����
        if len(visited_list) == 0:
            continue
        # ���Լ���i���û��ķ�����Ŀ��ȥ��
        test_list = list(testData[testData['user'] == user_i]['item'].drop_duplicates())
        # ���Լ���i���û��ķ�����Ŀ��ȥ�����û���ѵ�����ѷ��ʵ���Ŀ
        test_list = list(set(test_list) - set(test_list).intersection(set(visited_list)))
        # ���Լ���i���û��ķ�����ĿΪ�գ�����
        if len(test_list) == 0:
            continue
        # ���ɲ��Լ���i���û�δ���ʵ���Ŀ:���ֶ�
        poss = {}
        for item in all_item:
            if item in visited_list:
                continue
            else:
                poss[item] = np.dot(P[user_i, :], Q[:, item])
        # ���ɲ��Լ���i���û����Ƽ��б�
        ranked_list, test_score = topK(poss, k)
        # ���в��Լ���i���û�������Ŀ���б�
        h = list(set(test_list).intersection(set(ranked_list)))
        Hits += len(h)
        for item in test_list:
            for i in range(len(ranked_list)):
                if item == ranked_list[i]:
                    MRR += 1 / (i+1)
                    NDCG += 1 / (math.log2(i+1+1))
                else:
                    continue
    HR = Hits / test_lenght
    MRR /= test_lenght
    NDCG /= test_lenght
    # ����ʱ��
    et = time.time()
    print("HR@10:%.4f\nMRR@10:%.4f\nNDCG@10:%.4f\nTotal time:%.4f" % (HR, MRR, NDCG, et-st))


# �����Ƽ�
def offline_rec(rec_k=10, train_k=30, steps=10):
    data, train_data, test_data = split_data()
    all_user, all_item = getData(data)
    rating = getUserItem(train_data, all_user, all_item)
    P, Q = train(rating, train_k, steps)
    # ��¼��ǰʱ��
    date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for user_i in all_user:
        # ���Լ���i���û���ѵ�����ѷ��ʵ���Ŀ
        visited_list = list(data[data['user'] == user_i]['item'])
        # û��ѵ�����ݣ�����
        if len(visited_list) == 0:
            continue
        # ���ɲ��Լ���i���û�δ���ʵ���Ŀ:���ֶ�
        poss = {}
        for item in all_item:
            if item in visited_list:
                continue
            else:
                poss[item] = np.dot(P[user_i, :], Q[:, item])
        # ���ɲ��Լ���i���û����Ƽ��б��������ݿ�
        ranked_list, test_score = topK(poss, rec_k)
        try:
            models.Rec.objects.filter(user_id=user_i).update(item_ids=str(ranked_list), update_time=date_time)
        except:
            models.Rec.objects.create(user_id=user_i, item_ids=str(ranked_list), create_time=date_time, update_time=date_time)
