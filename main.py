import utils
import recommend
import numpy as np

test = utils.read_test()
users, movies, ratings = test['userId'], test['movieId'], test['rating']

recommend.recommend(12,1)



# 开始预测
preds = []
for i in range(len(test)):
    print('%d/%d...' % (i+1, len(test)))
    preds.append(recommend.recommend_minihash(users[i], movies[i]))

SSE = np.sum(np.square(preds - ratings))
print(SSE)