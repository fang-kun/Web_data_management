import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import mean_squared_error

# 读取movies.dat文件
movies_df = pd.read_csv('../ml-1m/movies.dat', sep='::', engine='python', header=None,
                        names=['MovieID', 'Title', 'Genres'], encoding='latin1')

# 读取ratings.dat文件
ratings_df = pd.read_csv('../ml-1m/ratings.dat', sep='::', engine='python', header=None,
                         names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin1')

# 获取互动次数最多的前100部电影
top_100_movies = ratings_df['MovieID'].value_counts().head(100).index.tolist()

# 获取与这些电影互动至少20次的用户
user_interactions = ratings_df[ratings_df['MovieID'].isin(top_100_movies)]['UserID'].value_counts()
selected_users = user_interactions[user_interactions >= 20].index.tolist()
random_selected_users = np.random.choice(selected_users, size=1000, replace=False)

# 根据选定的用户和电影筛选数据
selected_ratings = ratings_df[
    (ratings_df['MovieID'].isin(top_100_movies)) & (ratings_df['UserID'].isin(random_selected_users))]

# 分割数据集为训练集和测试集
train_data, test_data = train_test_split(selected_ratings, test_size=0.1, random_state=42)

# 保存训练集和测试集
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')


def build_slope_one(train_data):
    freq = defaultdict(int)
    diff = defaultdict(float)
    for _, row in train_data.iterrows():
        user, item, rating = row['UserID'], row['MovieID'], row['Rating']
        freq[item] += 1
        diff[item] += rating

    for item1 in diff:
        for item2 in diff:
            if item1 != item2:
                diff[item1] -= diff[item2] / freq[item2]

    return diff, freq


def predict_slope_one(user, item, diff, freq):
    pred = 0.0
    if item in diff and item in freq:
        pred = diff[item] / freq[item]
    return pred


diff, freq = build_slope_one(train_data)

predictions = []
actual_ratings = []

for _, row in test_data.iterrows():
    user, item, rating = row['UserID'], row['MovieID'], row['Rating']
    pred_rating = predict_slope_one(user, item, diff, freq)
    predictions.append(pred_rating)
    actual_ratings.append(rating)


def recall_at_k(predictions, actual_ratings, k):
    top_k_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:k]
    top_k_actual_ratings = [actual_ratings[i] for i in top_k_indices]
    recall = sum(top_k_actual_ratings) / sum(actual_ratings)
    return recall


rmse = mean_squared_error(actual_ratings, predictions, squared=False)
print('RMSE:', rmse)

k = 10  # 召回率的计算中，选择前k个推荐结果进行评估
recall = recall_at_k(predictions, actual_ratings, k)
print('Recall@{}: {}'.format(k, recall))
