import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from collections import defaultdict

def find_new_frequent_items(movies_like_by_user,frequent_of_k,min_support):
    """
        movies_like_by_user:每一个人喜欢电影的集合,也就是前面的like_by_user
        frequent_of_k：超集，也就是前面例子图中的L1，L2等等
        min_support:最小的支持度
    """
    counts = defaultdict(int)
    # 获得用户喜欢的movies的集合
    for user,movie_ids in movies_like_by_user.items():
        # 遍历超集中间的数据项
        for itemset in frequent_of_k:
            # 如数据项在用户的movie集合中，则代表用户同时喜欢这几部电影
            if itemset.issubset(movie_ids):
                # 遍历出现在movie集合但是没有出现在数据项中间的数据
                for other_movie in movie_ids - itemset:
                    # current_superset为数据项和other_movie的并集
                    current_superset = itemset | frozenset((other_movie,))
                    counts[current_superset] += 1
	# 去除support小于min_support的，返回key为数据项，value为support的集合
    return dict([(itemset,support) for itemset,support in counts.items() if support >= min_support])


if __name__ == '__main__':
    all_ratings = pd.read_csv('../ml-1m/ratings.dat', header=None, delimiter='::', index_col=False, engine='python', names=['userId', 'movieId', 'rating'])
    all_ratings["like"] = all_ratings["rating"] >= 4

    train_num = 200
    # 训练数据
    train_ratings = all_ratings[all_ratings['userId'].isin(range(train_num))]

    like_ratings = train_ratings[train_ratings["like"] == True]
    like_by_user = dict((k, frozenset(v.values)) for k, v in like_ratings.groupby("userId")["movieId"])
    num_like_of_movie = like_ratings[["movieId", "like"]].groupby("movieId").sum()

    # frequent_itemsets是一个字典，key为K项值，value为也为一个字典
    frequent_itemsets = {}
    min_support = 50
    # first step 步骤一：生成初始的频繁数据集
    frequent_itemsets[1] = dict((frozenset((movie_id,)), row["like"])
                                for movie_id, row in num_like_of_movie.iterrows()
                                if row["like"] > min_support)
    for k in range(2,4):
        current_set = find_new_frequent_items(like_by_user, frequent_itemsets[k - 1], min_support)
        if len(current_set) == 0:
            break
        else:
            frequent_itemsets[k] = current_set
    # 删除第一项（也就是k=1的项）
    del frequent_itemsets[1]

    rules = []
    for k, item_counts in frequent_itemsets.items():
        # k代表项数，item_counts代表里面的项
        for item_set in item_counts.keys():
            for item in item_set:
                premise = item_set - set((item,))
                rules.append((premise, item))

    # 得到每一条规则在训练集中的应验的次数
    # 应验
    right_rule = defaultdict(int)
    # 没有应验
    out_rule = defaultdict(int)

    for user, movies in like_by_user.items():
        for rule in rules:
            # premise,item代表购买了premise就会购买item
            premise, item = rule
            if premise.issubset(movies):
                if item in movies:
                    right_rule[rule] += 1
                else:
                    out_rule[rule] += 1

    # 计算每一条规则的置信度
    rule_confidence = {rule: right_rule[rule] / float(right_rule[rule] + out_rule[rule]) for rule in rules}
    from operator import itemgetter

    # 进行从大到小排序
    sort_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)

    # 计算X在训练集中出现的次数
    item_num = defaultdict(int)
    for user, movies in like_by_user.items():
        for rule in rules:
            # item 代表的就是X
            premise, item = rule
            if item in movies:
                item_num[rule] += 1

    # 计算P(X) item_num[rule]代表的就是P(X)
    item_num = {k: v / len(like_by_user) for k, v in item_num.items()}

    # 计算每一条规则的Lift
    rule_lift = {rule: (right_rule[rule] / (float(right_rule[rule] + out_rule[rule]))) / item_num[rule] for rule in
                 rules}
    from operator import itemgetter

    # 进行排序
    sort_lift = sorted(rule_lift.items(), key=itemgetter(1), reverse=True)

    # 去除训练使用的数据集得到测试集
    ratings_test = all_ratings.drop(train_ratings.index)
    # 去除测试集中unlike数据
    like_ratings_test = ratings_test[ratings_test["like"]]
    user_like_test = dict((k, frozenset(v.values)) for k, v in like_ratings_test.groupby("userId")["movieId"])

    # 应验的次数
    right_rule = 0
    # 没有应验的次数
    out_rule = 0
    for movies in user_like_test.values():
        if (sort_lift[0][0][0].issubset(movies)):
            if (sort_lift[0][0][1] in movies):
                right_rule += 1
            else:
                out_rule += 1
    print("召回率为：{}".format(right_rule / (right_rule + out_rule)))
