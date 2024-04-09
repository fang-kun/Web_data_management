import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取movies.dat文件
movies_df = pd.read_csv('../ml-1m/movies.dat', sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin1')

# 读取ratings.dat文件
ratings_df = pd.read_csv('../ml-1m/ratings.dat', sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin1')

# 获取互动次数最多的前100部电影
top_100_movies = ratings_df['MovieID'].value_counts().head(100).index.tolist()

# 获取与这些电影互动至少20次的用户
user_interactions = ratings_df[ratings_df['MovieID'].isin(top_100_movies)]['UserID'].value_counts()
selected_users = user_interactions[user_interactions >= 20].index.tolist()
random_selected_users = np.random.choice(selected_users, size=1000, replace=False)

# 根据选定的用户和电影筛选数据
selected_ratings = ratings_df[(ratings_df['MovieID'].isin(top_100_movies)) & (ratings_df['UserID'].isin(random_selected_users))]

# 分割数据集为训练集和测试集
train_data, test_data = train_test_split(selected_ratings, test_size=0.1, random_state=42)

# 保存训练集和测试集
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# 合并训练集和测试集数据
combined_df = pd.concat([train_data, test_data])

# 使用pivot_table函数生成评分矩阵
rating_matrix = combined_df.pivot_table(index='UserID', columns='MovieID', values='Rating')

# # 评分
# r_ratings = np.random.randint(1, 6, size=rating_matrix.shape)
#
# # 计算随机推荐的RMSE
# random_rmse = np.sqrt(((rating_matrix - r_ratings) ** 2).mean().mean())
#
# print("随机推荐的RMSE：", random_rmse)

print(rating_matrix)