import numpy as np
import pandas as pd
import math
import utils

K = 100
pre =True
n=20

def recommend(userid, movieid):
    df_train = utils.read_train()
    #构建用户特征矩阵

    feature = df_train.pivot_table(index=['movieId'], columns=['userId'], values='rating').reset_index(drop=True)
    feature.fillna(0, inplace=True)
    feature.index = np.array(sorted(df_train['movieId'].unique()))



    sim_dict = {i: feature[userid].corr(feature[i], method='pearson') for i in range(1, 671 + 1)}

    sorted_sim = sorted(sim_dict.items(),key = lambda kv:(kv[1], kv[0]),reverse=True)
    topK_id = [sorted_sim[i][0] for i in range(K)]
    """

    up_sum = 0
    down_sum = 0
    mean = 0
    for i in range(K):
        up_sum+=sim_dict[topK_id[i]]*feature[topK_id[i]][movieid]
        down_sum+=sim_dict[topK_id[i]]
        mean+=feature[topK_id[i]][movieid]
    return mean/K#up_sum/down_sum"""
    # 取K个最相似的用户
    topK_matrix = feature[topK_id]
    if pre:
        # 直接预测评分模式
        x = topK_matrix.loc[movieid]
        pred_i = np.mean(x[x != 0])
        return pred_i
    pred_dict = {}
    for i in range(len(feature)):
        x = topK_matrix.iloc[i]#取topk用户电影信息，列
        if len(x[x != 0]) > 10:#大于10人才计算此电影
            pred_i = np.mean(x[x != 0])
            pred_dict[i] = 0 if np.isnan(pred_i) else pred_i#给出此电影评分/推荐分数
        else:
            pred_dict[i] = 0  # 不考虑

    # 取前n个电影推荐
    sorted_pred = sorted(pred_dict.items(), key=lambda d: d[1], reverse=True)#排序电影
    pred = sorted_pred[:n]
    print('User %d, the top %d recommendations:' % (userid, n))
    print('-----------------------------------------------------------')
    result = 0
    for i in range(n):
        id, score = pred[i]
        print('%6d -- %.4f' % (feature.index[id], score))

def recommend_minihash(userid, movieid):
    df_train = utils.read_train()

    nfuncs = 1000  # 映射函数数量
    # 是minhash模式，需要额外生成一个对rating进行01化的矩阵
    binary_data = df_train.copy(deep=True)
    binary_data.loc[binary_data['rating'] < 2.6, 'rating'] = 0
    binary_data.loc[binary_data['rating'] > 2.9, 'rating'] = 1

    # 构建用户特征矩阵
    feature = df_train.pivot_table(index=['movieId'], columns=['userId'], values='rating').reset_index(drop=True)
    feature.fillna(0, inplace=True)
    feature.index = np.array(sorted(df_train['movieId'].unique()))

    # 根据随机生成的nfuncs个映射函数生成哈希签名矩阵
    users_num = len(feature.columns)
    movies_num = len(feature[1])

    sig_matrix = np.zeros((nfuncs, users_num))
    for i in range(nfuncs):
        func = list(range(1, movies_num + 1))
        np.random.shuffle(func)  # permutation π
        shuffled_matrix = feature.reindex(func)
        s = set(range(users_num))  # 记录对于每个func，user是否找到第一个1的集合，当user找到了则从集合中弹出

        sig_i = np.zeros(users_num)
        for j in range(movies_num):
            row = np.array(shuffled_matrix.iloc[j])
            for r in range(users_num):
                if row[r] and r in s:
                    s.remove(r)
                    sig_i[r] = j + 1
            if not s:
                break

        sig_matrix[i] = sig_i  # 更新签名矩阵的第i行

    sig_matrix = pd.DataFrame(sig_matrix)
    sig_matrix.columns = list(range(1, users_num + 1))
    # print(sig_matrix)
    # 使用jaccard系数计算用户之间的相似度
    sim_dict = {i: np.sum(sig_matrix[userid] == sig_matrix[i]) / nfuncs for i in range(1, users_num + 1)}
    sorted_sim = sorted(sim_dict.items(), key=lambda d: d[1], reverse=True)

    # 取K个最相似的用户
    topK_id = [sorted_sim[i][0] for i in range(K)]
    topK_matrix = feature[topK_id]

    if pre:
        # 直接预测评分模式
        x = topK_matrix.loc[movieid]
        pred_i = np.mean(x[x != 0])
        return pred_i

    pred_dict = {}
    for i in range(len(feature)):
        x = topK_matrix.iloc[i]
        if len(x[x != 0]) > 15:
            pred_i = np.mean(x[x != 0])  # 去掉里面的0项
            pred_dict[i] = 0 if np.isnan(pred_i) else pred_i
        else:
            pred_dict[i] = 0  # 不考虑

    # 取前n个电影推荐
    sorted_pred = sorted(pred_dict.items(), key=lambda d: d[1], reverse=True)
    pred = sorted_pred[:n]
    print('User %d, the top %d recommendations are shown below:' % (userid, n))
    print('-----------------------------------------------------------')
    result = 0
    for i in range(n):
        id, score = pred[i]
        print('%6d | %.4f' % (feature.index[id], score))













