import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = \
            self.prepare_dicts(self.user_item_matrix)

        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        # self.item_id_to_ctm = #your_code

        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        """Подготавливает user_item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)

        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=200, regularization=0.01, iterations=20, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        # находим top-N покупок user
        top = self.top_purchases[self.top_purchases['user_id'] == user].head(N).item_id.tolist()

        # для каждого из top-N покупок user находим 1 наиболее ближайший товар и делаем рекомендации
        recs = []
        for item_id in top:
            rec = [self.id_to_itemid[i[0]] for i in self.model.similar_items(self.itemid_to_id[item_id], N=3)]
            rec_id = [i for i in rec if i != 999999]  # отсеиваем 999999
            similar_item != rec_id[1]  # отсеиваем сам товар
            recs.append(similar_item)

        # только уникальные рекоммендации
        recs = list(set(recs))

        return recs

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        """top-1 от каждого из N-ти похожих юзеров"""
        # находим N наиболее похожих юзеров
        users_id = [self.id_to_userid[i[0]] for i in self.model.similar_users(self.userid_to_id[user], N=N + 1)]
        similar_users = [i for i in users_id if i != user]

        # N покупок по 1 самой популярной от каждого из N ближайших юзеров
        top_purchases_n_users = self.top_purchases.groupby('user_id').head(1)
        top_purchases_n_users = top_purchases_n_users[top_purchases_n_users['user_id'].isin(similar_users)]
        top_purchases_n_users.sort_values('user_id', ascending=False, inplace=True)
        recs = top_purchases_n_users.item_id.tolist()

        return recs