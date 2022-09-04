# -*- coding: utf-8 -*-
import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import heapq
import numpy as np
import math


#������Ŀtop_K����
def topK_scores(test, predict, topk, user_count, item_count):

    PrecisionSum = np.zeros(topk+1)
    RecallSum = np.zeros(topk+1)
    F1Sum = np.zeros(topk+1)
    NDCGSum = np.zeros(topk+1)
    OneCallSum = np.zeros(topk+1)
    DCGbest = np.zeros(topk+1)
    MRRSum = 0
    MAPSum = 0
    total_test_data_count = 0
    for k in range(1, topk+1):
        DCGbest[k] = DCGbest[k - 1]
        DCGbest[k] += 1.0 / math.log(k + 1)
    for i in range(user_count):
        user_test = []
        user_predict = []
        test_data_size = 0
        for j in range(item_count):
            if test[i * item_count + j] == 1.0:
                test_data_size += 1
            user_test.append(test[i * item_count + j])
            user_predict.append(predict[i * item_count + j])
        if test_data_size == 0:
            continue
        else:
            total_test_data_count += 1
        predict_max_num_index_list = map(user_predict.index, heapq.nlargest(topk, user_predict))
        predict_max_num_index_list = list(predict_max_num_index_list)
        hit_sum = 0
        DCG = np.zeros(topk + 1)
        DCGbest2 = np.zeros(topk + 1)
        for k in range(1, topk + 1):
            DCG[k] = DCG[k - 1]
            item_id = predict_max_num_index_list[k - 1]
            if user_test[item_id] == 1:
                hit_sum += 1
                DCG[k] += 1 / math.log(k + 1)
            # precision, recall, F1, 1-call
            prec = float(hit_sum / k)
            rec = float(hit_sum / test_data_size)
            f1 = 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            PrecisionSum[k] += float(prec)
            RecallSum[k] += float(rec)
            F1Sum[k] += float(f1)
            if test_data_size >= k:
                DCGbest2[k] = DCGbest[k]
            else:
                DCGbest2[k] = DCGbest2[k-1]
            NDCGSum[k] += DCG[k] / DCGbest2[k]
            if hit_sum > 0:
                OneCallSum[k] += 1
            else:
                OneCallSum[k] += 0
        # MRR
        p = 1
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                break
            p += 1
        MRRSum += 1 / float(p)
        # MAP
        p = 1
        AP = 0.0
        hit_before = 0
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                AP += 1 / float(p) * (hit_before + 1)
                hit_before += 1
            p += 1
        MAPSum += AP / test_data_size
    print('MAP:', MAPSum / total_test_data_count)
    print('MRR:', MRRSum / total_test_data_count)
    print('Prec@5:', PrecisionSum[4] / total_test_data_count)
    print('Rec@5:', RecallSum[4] / total_test_data_count)
    print('F1@5:', F1Sum[4] / total_test_data_count)
    print('NDCG@5:', NDCGSum[4] / total_test_data_count)
    print('1-call@5:', OneCallSum[4] / total_test_data_count)
    return


class BPR:
    #�û���
    user_count = 943
    #��Ŀ��
    item_count = 1682
    #k������,k��
    latent_factors = 20
    #������
    lr = 0.01
    #������
    reg = 0.01
    #ѵ������
    train_count = 10000
    #ѵ����
    train_data_path = 'train.txt'
    #���Լ�
    test_data_path = 'test.txt'
    #U-I�Ĵ�С
    size_u_i = user_count * item_count
    # ����趨��U��V����(����ʽ�е�Wuk��Hik)����
    U = np.random.rand(user_count, latent_factors) * 0.01 #��С����ν
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01
    #����һ���û���*��Ŀ����С��ȫ0����
    test_data = np.zeros((user_count, item_count))
    print("test_data_type",type(test_data))
    #����һ��һά��ȫ0����
    test = np.zeros(size_u_i)
    #������һ��һά��ȫ0����
    predict_ = np.zeros(size_u_i)

    # ����˵����ͨ���ļ�·������ȡU-I����
    # Paramaters:
    #     ����Ҫ������ļ�·��path
    # Returns:
    #     ���һ���ֵ�user_ratings�������û�-��Ŀ�ļ�ֵ��
    def load_data(self, path):
        user_ratings = defaultdict(set)
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
        return user_ratings

    # ����˵����ͨ���ļ�·������ȡ���Լ�����
    # Paramaters��
    #     ���Լ��ļ�·��path
    # Returns:
    #     ���һ��numpy.ndarray�ļ���nά���飩test_data,���аѺ��з�����Ϣ��������Ϊ1
    #��ȡ���Լ������־���
    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1


    # ����˵������ѵ���������ֵ䴦��ͨ�����ѡȡ�����û���������Ϊ��������Ԫ�飬���·ֽ�����������
    # Parameters��
    #     ����Ҫ�����ѵ�����û���Ŀ�ֵ�
    # Returns��
    #     �Էֽ������������Լ�ƫ�þ���ֱ����
    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # �����ȡһ���û�
            u = random.randint(1, self.user_count) #�ҵ�һ��user
            # ѵ�����Ͳ��Լ������ڲ���ȫ��һ����,����train��948,��test���Ϊ943
            if u not in user_ratings_train.keys():
                continue
            # ���û���U-I�����ѡȡ1��Item
            i = random.sample(user_ratings_train[u], 1)[0] #�ҵ�һ��item��������
            # ���ѡȡһ���û�uû�����ֵ���Ŀ
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count) #�ҵ�һ��item��û�б�����
            #����һ����Ԫ�飨uesr,item_have_score,item_no_score)
            # python�е�ȡֵ��0��ʼ
            u = u - 1
            i = i - 1
            j = j - 1
            #BPR
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # ����2������
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # ����ƫ����
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])


    # ����˵����ͨ������ֽ����û���Ŀ����õ�Ԥ�����predict
    # Parameters:
    #     ����ֱ����û���Ŀ����
    # Returns��
    #     �����˺��Ԥ����󣬼�������Ҫ�����־���
    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    #������
    def main(self):
        #��ȡU-I��{1:{2,5,1,2}....}����
        user_ratings_train = self.load_data(self.train_data_path)
        #��ȡ���Լ������־���
        self.load_test_data(self.test_data_path)
        #��test_data������ƽ
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        #ѵ��
        for i in range(self.train_count):
            self.train(user_ratings_train)  #ѵ��10000�����
        predict_matrix = self.predict(self.U, self.V) #��ѵ����ɵľ���Ȼ�
        # Ԥ��
        self.predict_ = predict_matrix.getA().reshape(-1)  #.getA()������������ת��Ϊndarray���͵ı���
        print("predict_new",self.predict_)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)

    # ����˵�����Խ���������������û��Ѿ������������û���Ŀ�����޳���ֻ����û�в����û���Ŀ�Ľ���������
    # Paramaters:
    #     �����û���Ŀ�ֵ伯���Լ�һά��Ԥ�������Ŀ����
    # Returns:
    #     ����������Ԥ������һά��Ԥ�����
def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict
