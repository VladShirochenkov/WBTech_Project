import os
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from .logger import setup_logger
from .config import Config

class RecommendationCluster:
    def __init__(self):
        # Настройка логгера
        self.logger = setup_logger()
        self.logger.info("Инициализация RecommendationCluster...")
        
        # Загрузка конфигурации
        self.config = Config()
        
        # Параметры кластеризации из раздела cluster
        self.n_clusters_users = self.config.cluster.get('n_clusters_users', 24)
        self.random_state = self.config.cluster.get('random_state', 42)
        self.n_init = self.config.cluster.get('n_init', 10)
        
        # Пути к данным из раздела cluster
        self.data_path = os.path.join(os.path.dirname(__file__), '..', self.config.cluster.get('data_path', 'data/processed/'))
        self.users_emb_path = os.path.join(self.data_path, self.config.cluster.get('users_emb_file', 'users_emb_full.parquet'))
        self.first_train_path = os.path.join(self.data_path, self.config.cluster.get('first_train_file', 'first_train_2.parquet'))
        self.recommendation_files = {
            'val': os.path.join(self.data_path, self.config.cluster.get('recommendations', {}).get('val_file', 'recommendations_val.parquet')),
            'test': os.path.join(self.data_path, self.config.cluster.get('recommendations', {}).get('test_file', 'recommendations_test.parquet'))
        }
        
        tqdm.pandas()
        
        self.load_data()
        self.cluster_users()
        self.logger.info("Инициализация RecommendationCluster завершена.")
    
    def load_data(self):
        self.logger.info(f"Загрузка эмбеддингов пользователей из {self.users_emb_path}...")
        self.users_emb = pd.read_parquet(self.users_emb_path)
        self.logger.info(f"Эмбеддинги пользователей загружены, форма данных: {self.users_emb.shape}")
        
        self.logger.info(f"Загрузка тренировочных данных из {self.first_train_path}...")
        self.first_train = pd.read_parquet(self.first_train_path)
        self.logger.info(f"Тренировочные данные загружены, форма данных: {self.first_train.shape}")
    
    def cluster_users(self):
        self.logger.info("Кластеризация пользователей с помощью KMeans...")
        self.kmeans_users = KMeans(
            n_clusters=self.n_clusters_users, 
            random_state=self.random_state, 
            n_init=self.n_init
        )
        # Убедимся, что эмбеддинги пользователей содержат только числовые значения
        self.users_emb['user_cluster'] = self.kmeans_users.fit_predict(self.users_emb.values)
        self.logger.info("Кластеризация завершена. Метки кластеров присвоены пользователям.")
    
    def compute_normalized_cooccurrence(self, row):
        candidate_item = row['item_id']
        user_id = row['user_id']
        user_cluster = row['user_cluster']

        interacted_items = self.cluster_user_items_dict.get((user_cluster, user_id), set())
        candidate_users = self.cluster_item_users_dict.get((user_cluster, candidate_item), set())
        candidate_popularity = len(candidate_users)

        if candidate_popularity == 0:
            return 0.0

        cooccurrences = []
        for item in interacted_items:
            users = self.cluster_item_users_dict.get((user_cluster, item), set())
            co_users = candidate_users.intersection(users)
            co_count = len(co_users)
            normalized = co_count / candidate_popularity
            cooccurrences.append(normalized)
        
        return sum(cooccurrences) / len(cooccurrences) if cooccurrences else 0.0

    def compute_stats(self, dataset='val'):
        if dataset not in ['val', 'test']:
            self.logger.error(f"Неверный параметр dataset: {dataset}. Ожидалось 'val' или 'test'.")
            raise ValueError("`dataset` должно быть либо 'val', либо 'test'.")

        path = self.recommendation_files[dataset]
        self.logger.info(f"Обработка набора данных: {dataset} | Файл: {path}")

        recommendations_df = pd.read_parquet(path)
        self.logger.info(f"Рекомендации загружены, форма данных: {recommendations_df.shape}")

        users_clusters = pd.DataFrame({
            'user_id': self.users_emb.index,
            'user_cluster': self.users_emb['user_cluster']
        })
        self.logger.info("Кластеры пользователей подготовлены для объединения.")

        first_train_with_user_cluster = self.first_train.merge(
            users_clusters, 
            on='user_id', 
            how='left'
        )
        recommendations_with_user_cluster = recommendations_df.merge(
            users_clusters, 
            on='user_id', 
            how='left'
        )
        self.logger.info("Кластеры пользователей объединены с тренировочными и рекомендационными данными.")

        self.cluster_user_items_dict = first_train_with_user_cluster.groupby(['user_cluster', 'user_id'])['item_id'].apply(set).to_dict()
        self.cluster_item_users_dict = first_train_with_user_cluster.groupby(['user_cluster', 'item_id'])['user_id'].apply(set).to_dict()
        self.logger.info("Созданы словари cluster_user_items_dict и cluster_item_users_dict.")

        self.logger.info("Добавление столбца 'interacted_items'...")
        recommendations_with_user_cluster['interacted_items'] = recommendations_with_user_cluster.progress_apply(
            lambda row: self.cluster_user_items_dict.get((row['user_cluster'], row['user_id']), set()), 
            axis=1
        )

        self.logger.info("Вычисление 'cluster_avg_normalized_cooccurrence'...")
        recommendations_with_user_cluster['cluster_avg_normalized_cooccurrence'] = recommendations_with_user_cluster.progress_apply(
            self.compute_normalized_cooccurrence, 
            axis=1
        )

        self.logger.info("Вычисление популярности предметов внутри кластеров...")
        popularity = first_train_with_user_cluster.groupby(['user_cluster', 'item_id']).size().reset_index(name='item_cluster_popularity')
        cluster_total_popularity = popularity.groupby('user_cluster')['item_cluster_popularity'].sum().reset_index(name='cluster_total_popularity')
        popularity_normalized = popularity.merge(cluster_total_popularity, on='user_cluster', how='left')
        popularity_normalized['item_cluster_popularity_normalized'] = popularity_normalized.apply(
            lambda row: row['item_cluster_popularity'] / row['cluster_total_popularity'] if row['cluster_total_popularity'] > 0 else 0, 
            axis=1
        )
        self.logger.info("Нормализованная популярность предметов вычислена.")

        recommendations_with_features = recommendations_with_user_cluster.merge(
            popularity_normalized[['user_cluster', 'item_id', 'item_cluster_popularity_normalized']],
            on=['user_cluster', 'item_id'], 
            how='left'
        )
        recommendations_with_features['item_cluster_popularity_normalized'] = recommendations_with_features['item_cluster_popularity_normalized'].fillna(0)
        self.logger.info("Нормализованная популярность объединена с рекомендациями.")

        recommendations_with_features = recommendations_with_features.drop(['user_cluster', 'interacted_items'], axis=1)
        self.logger.info("Удалены вспомогательные столбцы.")

        # Сохранение расширенных рекомендаций обратно в файл
        recommendations_with_features.to_parquet(path)
        self.logger.info(f"Расширенные рекомендации сохранены в {path}.\n")

