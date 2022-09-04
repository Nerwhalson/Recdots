import numpy as np
import pandas as pd
import os, time, math, datetime

REC_PATH = os.path.join(os.getcwd(), "utils", "rec_models", "save")
from .. import models


# 参数说明：
#     R:用户-物品对应的共现矩阵 m*n
#     P:用户因子矩阵 m*k
#     Q:物品因子矩阵 k*n
#     K:隐向量的维度 
#     steps:最大迭代次数


# 将矩阵R分解成P,Q
def matrix_factorization(R, P, Q, K, steps, alpha=0.05, Lambda=0.002):
    # 总时长
    sum_st = 0
    # 前一次的损失大小
    e_old = 0
    # 程序结束的标识
    flag = 1
    # 梯度下降结束条件1：满足最大迭代次数
    for step in range(steps):
        # 每次跌代开始的时间
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
        # 第一次迭代不执行前后损失之差
        if step == 0:
            e_old = e_new
            continue
        # 梯度下降结束条件2：loss过小，跳出
        if e_new < 1e-3:
            flag = 2
            break
        # 梯度下降结束条件3：前后loss之差过小，跳出
        if (e_old - e_new) < 1e-10:
            flag = 3
            break
        else:
            e_old = e_new
    print(f'--------Summary---------\nThe type of jump out:{flag}\nTotal steps:{step+1}\nTotal time:{sum_st}\n'
          f'Average time:{sum_st / (step+1)}\nThe e is :{e_new}')
    return P, Q


# 分割数据集成训练集、测试集
def split_data():
    # 读取原始数据
    data = pd.DataFrame(models.Behavior.objects.values())
    data = data[data['behave_type'] == 'Score']
    rating = data[['user_id', 'item_id', 'score', 'update_time']]
    rating['update_time'].apply(lambda x: x.to_pydatetime().timestamp())
    rating = rating.rename(columns={'user_id': 'user', 'item_id': 'item', 'score': 'score', 'update_time': 'time'})
    # 按照时间顺序排序
    rating.sort_values(by=['time'], axis=0, inplace=True)
    # 按照时间顺序值8:2，确定边界线
    boundary1 = rating['time'].quantile(0.8)
    # boundary2 = rating['time'].quantile(0.8)

    # 按时间分界点切分数据，生成训练集
    train = rating[rating['time'] < boundary1]
    # 训练集按用户、时间顺序排序
    train.sort_values(by=['user', 'time'], axis=0, inplace=True)

    # 按时间分界点切分数据，生成测试集
    test = rating[rating['time'] >= boundary1]
    # 验证集按用户、时间顺序排序
    test.sort_values(by=['user', 'time'], axis=0, inplace=True)
    
    data = pd.concat([train, test])

    # 将训练集、测试集写入文件中
    train.to_csv(os.path.join(REC_PATH, "Train.txt"), sep=',', index=False, header=None)
    test.to_csv(os.path.join(REC_PATH, "Test.txt"), sep=',', index=False, header=None)

    return data, train, test


# 获取本地数据
def getData(data):
    # 总用户数量
    all_user = np.unique(data['user'])
    # 总项目数量
    all_item = np.unique(data['item'])
    return all_user, all_item


# 生成用户-物品矩阵并保存到本地文件中
def getUserItem(train_data, all_user, all_item):
    train_data.sort_values(by=['user', 'item'], axis=0, inplace=True)
    # 用户-项目共现矩阵行数
    num_user = np.max(all_user)+1
    # 用户-项目共现矩阵列数
    num_item = np.max(all_item)+1
    # 用户-项目共现矩阵初始化
    rating_mat = np.zeros([num_user, num_item], dtype=int)
    # 用户-项目共现矩阵赋值
    for i in range(len(train_data)):
        user = train_data.iloc[i]['user']
        item = train_data.iloc[i]['item']
        score = train_data.iloc[i]['score']
        rating_mat[user][item] = score
    # 保存用户-项目共现矩阵到文件
    np.savetxt(os.path.join(REC_PATH, "rating.txt"), rating_mat, fmt='%d', delimiter=',', newline='\n')
    print(f'generate rating matrix complete!')

    return rating_mat


# 生成topk推荐列表
def topK(dic, k):
    keys = []
    values = []
    for i in range(k):
        key, value = max(dic.items(), key=lambda x: x[1])
        keys.append(key)
        values.append(value)
        dic.pop(key)
    return keys, values


#	训练
def train(rating, K, steps):
    R = rating
    M = len(R)
    N = len(R[0])
    # 用户矩阵初始化
    P = np.random.normal(loc=0, scale=0.01, size=(M, K))
    # 项目矩阵初始化
    Q = np.random.normal(loc=0, scale=0.01, size=(K, N))
    P, Q = matrix_factorization(R, P, Q, K, steps)
    # 将P，Q保存到文件
    np.savetxt(os.path.join(REC_PATH, "MF_userMatrix.txt"), P, fmt="%.6f", delimiter=',', newline='\n')
    np.savetxt(os.path.join(REC_PATH, "MF_itemMatrix.txt"), Q, fmt="%.6f", delimiter=',', newline='\n')
    print("train complete!")

    return P, Q

# 测试
def test(trainData, testData, all_item, k):
    # 读取用户矩阵
    P = np.loadtxt(os.path.join(REC_PATH, "MF_userMatrix.txt"), delimiter=',', dtype=float)
    # 读取项目矩阵
    Q = np.loadtxt(os.path.join(REC_PATH, "MF_itemMatrix.txt"), delimiter=',', dtype=float)
    # 测试集中的用户集合
    testUser = np.unique(testData['user'])
    # 测试集的长度
    test_lenght = len(testData)

    Hits = 0
    MRR = 0
    NDCG= 0
    # 开始时间
    st = time.time()
    for user_i in testUser:
        # 测试集第i个用户在训练集已访问的项目
        visited_list = list(trainData[trainData['user'] == user_i]['item'])
        # 没有训练数据，跳过
        if len(visited_list) == 0:
            continue
        # 测试集第i个用户的访问项目并去重
        test_list = list(testData[testData['user'] == user_i]['item'].drop_duplicates())
        # 测试集第i个用户的访问项目中去除该用户在训练集已访问的项目
        test_list = list(set(test_list) - set(test_list).intersection(set(visited_list)))
        # 测试集第i个用户的访问项目为空，跳过
        if len(test_list) == 0:
            continue
        # 生成测试集第i个用户未访问的项目:评分对
        poss = {}
        for item in all_item:
            if item in visited_list:
                continue
            else:
                poss[item] = np.dot(P[user_i, :], Q[:, item])
        # 生成测试集第i个用户的推荐列表
        ranked_list, test_score = topK(poss, k)
        # 命中测试集第i个用户访问项目的列表
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
    # 结束时间
    et = time.time()
    print("HR@10:%.4f\nMRR@10:%.4f\nNDCG@10:%.4f\nTotal time:%.4f" % (HR, MRR, NDCG, et-st))


# 离线推荐
def offline_rec(rec_k=10, train_k=30, steps=10):
    data, train_data, test_data = split_data()
    all_user, all_item = getData(data)
    rating = getUserItem(train_data, all_user, all_item)
    P, Q = train(rating, train_k, steps)
    # 记录当前时间
    date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for user_i in all_user:
        # 测试集第i个用户在训练集已访问的项目
        visited_list = list(data[data['user'] == user_i]['item'])
        # 没有训练数据，跳过
        if len(visited_list) == 0:
            continue
        # 生成测试集第i个用户未访问的项目:评分对
        poss = {}
        for item in all_item:
            if item in visited_list:
                continue
            else:
                poss[item] = np.dot(P[user_i, :], Q[:, item])
        # 生成测试集第i个用户的推荐列表并更新数据库
        ranked_list, test_score = topK(poss, rec_k)
        try:
            models.Rec.objects.filter(user_id=user_i).update(item_ids=str(ranked_list), update_time=date_time)
        except:
            models.Rec.objects.create(user_id=user_i, item_ids=str(ranked_list), create_time=date_time, update_time=date_time)
