import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

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

train_data = pd.read_csv('train_data.csv')
X_train = train_data[['UserID', 'MovieID']].values
y_train = train_data['Rating'].values

mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
mlp.fit(X_train, y_train)

test_data = pd.read_csv('test_data.csv')
X_test = test_data[['UserID', 'MovieID']].values
y_true = test_data['Rating'].values

y_pred = mlp.predict(X_test)
rmse = mean_squared_error(y_true, y_pred, squared=False)
print('RMSE:', rmse)