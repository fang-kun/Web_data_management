import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from math import sqrt

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

def NaiveBayes_based_RS():
    # 加载训练集和测试集数据
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')

    # 准备特征和目标数据
    X_train = train_data[['UserID', 'MovieID']]
    y_train = train_data['Rating']
    X_test = test_data[['UserID', 'MovieID']]
    y_test = test_data['Rating']

    # 创建并训练Naive Bayes模型
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 进行预测并计算RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# 训练集作为评分矩阵 以便后续计算相似度矩阵
rating_matrix = train_data.pivot_table(index='UserID', columns='MovieID', values='Rating')


# User based CF
def user_similarity(rating_matrix):
    # 皮尔森算法计算用户之间的相似度
    similarity_matrix = rating_matrix.corr(method='pearson', min_periods=1)
    return similarity_matrix

def predict_user_cf(user_id,  rating_matrix, similarity_matrix, min_similar_users=2):
    if user_id in similarity_matrix.index:
        user_similarity = similarity_matrix.loc[user_id]
        user_ratings = rating_matrix.loc[user_id]

        similar_users = user_similarity[user_similarity > 0].index
        # 如果相似用户数量小于2，则预测评分返回0
        if len(similar_users) < min_similar_users:
            return 0.0  # 设置默认预测评分

        # 根据相似用户的评分和相似度进行加权平均 得出预测
        prediction = (user_similarity[similar_users] * (user_ratings.loc[similar_users] - user_ratings.mean())).sum() / \
                     user_similarity[similar_users].sum() + user_ratings.mean()
        return prediction
    else:
        return 0.0

def calculate_rmse(test_data, rating_matrix, similarity_matrix, cf_type):
    predictions = []

    for index, row in test_data.iterrows():
        user_id = row['UserID']
        # movie_id = row['MovieID']
        # actual_rating = row['Rating']

        if cf_type == 'user':
            prediction = predict_user_cf(user_id, rating_matrix, similarity_matrix)

        predictions.append(prediction)

    rmse = sqrt(mean_squared_error(test_data['Rating'], predictions))
    return rmse

def rmse_user():
    user_similarity_matrix = user_similarity(rating_matrix)
    rmse_user = calculate_rmse(test_data, rating_matrix, user_similarity_matrix, 'user')
    return rmse_user


if __name__ == '__main__':
    # 与User based CF比较
    rmse_bayes = NaiveBayes_based_RS()
    rmse_user = rmse_user()
    print("基于朴素贝叶斯的RMSE:", rmse_bayes)
    print("基于用户协同过滤的RMSE:", rmse_user)

    performance_gain = rmse_user - rmse_bayes
    percentage_gain = (performance_gain / rmse_user) * 100

    print("与协同过滤相比，朴素贝叶斯的性能提升： {:.2f}%".format(percentage_gain))