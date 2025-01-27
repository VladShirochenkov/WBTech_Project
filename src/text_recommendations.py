import pandas as pd
from sklearn.preprocessing import normalize
import faiss
import os
from .logger import setup_logger
from .config import Config

class RecommendationsText:
    def __init__(self):
        self.logger = setup_logger()
        self.config = Config()
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        self.load_data()
        self.prepare_embeddings()
        self.build_faiss_index()

    def load_data(self):
        try:
            self.logger.info("Загрузка данных для рекомендаций...")
            self.items = pd.read_parquet(os.path.join(self.data_dir, 'items_full.parquet'))
            self.first_train = pd.read_parquet(os.path.join(self.data_dir, 'first_train_2.parquet'))
            self.first_val = pd.read_parquet(os.path.join(self.data_dir, 'first_val_2.parquet'))
            self.warm_test = pd.read_parquet(os.path.join(self.data_dir, 'warm_test_2.parquet'))
            self.logger.info("Данные успешно загружены.")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def prepare_embeddings(self):
        try:
            self.logger.info("Подготовка эмбеддингов пользователей и товаров...")
            interaction_counts = self.first_train.groupby(['user_id', 'item_id']).size().reset_index(name='count')
            interaction_emb = interaction_counts.merge(self.items, on='item_id', how='left')

            embedding_cols = [col for col in self.items.columns if col.startswith('text_')]
            interaction_emb[embedding_cols] = interaction_emb[embedding_cols].multiply(interaction_emb['count'], axis=0)

            user_embeddings_sum = interaction_emb.groupby('user_id')[embedding_cols].sum()
            user_total_counts = interaction_emb.groupby('user_id')['count'].sum()
            user_embeddings = user_embeddings_sum.div(user_total_counts, axis=0).reset_index()

            self.user_ids = user_embeddings['user_id'].values
            self.user_emb_matrix = user_embeddings[embedding_cols].values

            self.item_ids = self.items['item_id'].values
            self.item_emb_matrix = self.items[embedding_cols].values

            self.user_emb_matrix_normalized = normalize(self.user_emb_matrix, axis=1)
            self.item_emb_matrix_normalized = normalize(self.item_emb_matrix, axis=1)

            self.dimension = self.item_emb_matrix_normalized.shape[1]
            self.embedding_cols = embedding_cols

            self.logger.info("Эмбеддинги успешно подготовлены.")
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке эмбеддингов: {e}")
            raise

    def build_faiss_index(self):
        try:
            self.logger.info("Построение FAISS индекса...")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.item_emb_matrix_normalized)
            self.logger.info(f"Количество добавленных в индекс векторов: {self.index.ntotal}")
        except Exception as e:
            self.logger.error(f"Ошибка при построении FAISS индекса: {e}")
            raise

    def generate_recommendations(self, user_ids_subset, output_file):
        try:
            self.logger.info(f"Генерация рекомендаций и сохранение в {output_file}...")
            user_embeddings_subset = self.user_emb_matrix_normalized[
                [i for i, uid in enumerate(self.user_ids) if uid in user_ids_subset]
            ]

            subset_user_ids = [uid for uid in self.user_ids if uid in user_ids_subset]

            top_n = 50
            D, I = self.index.search(user_embeddings_subset, top_n)
            index_to_item_id = {idx: item_id for idx, item_id in enumerate(self.item_ids)}

            recommendations = []
            for user_idx, user_id in enumerate(subset_user_ids):
                top_item_indices = I[user_idx]
                top_items = [index_to_item_id[idx] for idx in top_item_indices]
                for rank, item in enumerate(top_items, start=1):
                    recommendations.append({
                        'user_id': user_id,
                        'item_id': item,
                        'rank': rank
                        })

            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df.to_parquet(os.path.join(self.data_dir, output_file))
            self.logger.info(f"Рекомендации успешно сохранены в {output_file}.")
        except Exception as e:
            self.logger.error(f"Ошибка при генерации рекомендаций: {e}")
            raise

    def recos(self):
        # Генерация рекомендаций для валидации
        val_users = self.first_val['user_id'].unique()
        self.generate_recommendations(val_users, 'text_recommendations_val.parquet')

        # Генерация рекомендаций для теста
        test_users = self.warm_test['user_id'].unique()
        self.generate_recommendations(test_users, 'text_recommendations_test.parquet')

