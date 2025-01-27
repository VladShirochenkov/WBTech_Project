import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from pandarallel import pandarallel
from .config import Config
from .logger import setup_logger

class RecommendationStats:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()

        self.logger.info("Инициализация pandarallel...")
        pandarallel.initialize(progress_bar=True)
        self.logger.info("pandarallel инициализирован.")

        data_stats_config = self.config.data_recos_stats['input']
        self.items_full_path = os.path.join(os.path.dirname(__file__), '..', data_stats_config['items_full'])
        self.first_train_path = os.path.join(os.path.dirname(__file__), '..', data_stats_config['first_train'])
        self.recommendations_paths = {
            'val': os.path.join(os.path.dirname(__file__), '..', data_stats_config['recommendations']['val']),
            'test': os.path.join(os.path.dirname(__file__), '..', data_stats_config['recommendations']['test'])
        }

        self.load_data()

        self.prepare_mappings()
        
    def load_data(self):
        self.logger.info("Загрузка данных...")
        self.items = pd.read_parquet(self.items_full_path)
        self.first_train = pd.read_parquet(self.first_train_path)
        self.logger.info(f"Данные загружены: {len(self.items)} items, {len(self.first_train)} взаимодействий.")
        
    def prepare_mappings(self):
        self.logger.info("Подготовка маппингов и матриц...")
        # Маппинг item_id к индексу
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.items['item_id'])}
        
        # Подготовка текстовых эмбеддингов
        text_embedding_cols = [col for col in self.items.columns if col.startswith('text_')]
        text_emb_matrix = self.items[text_embedding_cols].values
        self.text_emb_normalized = normalize(text_emb_matrix, norm='l2', axis=1)
        
        # Пользовательские взаимодействия
        self.user_interactions = self.first_train.groupby('user_id')['item_id'].apply(list).to_dict()
        
        # Словарь: item_id -> set(user_id)
        item_users = self.first_train.groupby('item_id')['user_id'].apply(set).reset_index()
        self.item_users_dict = dict(zip(item_users['item_id'], item_users['user_id']))
        
        # Словарь: user_id -> set(item_id)
        user_items = self.first_train.groupby('user_id')['item_id'].apply(set).reset_index()
        user_items = user_items.rename(columns={'item_id': 'interacted_items'})
        self.user_items_df = user_items
        
        self.logger.info("Маппинги и матрицы подготовлены.")
        
    def compute_distance_stats(self, row):
        user_id = row['user_id']
        candidate_item_id = row['item_id']
        
        # Получение индекса кандидата
        candidate_idx = self.item_id_to_idx.get(candidate_item_id)
        if candidate_idx is None:
            self.logger.warning(f"Item_id {candidate_item_id} не найден в item_id_to_idx.")
            return pd.Series({'text_min_dist': np.nan})
        
        # Получение взаимодействованных item_ids пользователя
        interacted_item_ids = self.user_interactions.get(user_id, [])
        if not interacted_item_ids:
            return pd.Series({'text_min_dist': np.nan})

        # Преобразование item_ids во внутренние индексы
        interacted_indices = [self.item_id_to_idx.get(item_id) for item_id in interacted_item_ids]
        interacted_indices = [idx for idx in interacted_indices if idx is not None]
        
        if not interacted_indices:
            return pd.Series({'text_min_dist': np.nan})
        
        # Получение эмбеддингов
        candidate_text_emb = self.text_emb_normalized[candidate_idx].reshape(1, -1)
        interacted_text_emb = self.text_emb_normalized[interacted_indices]
        
        # Вычисление расстояний
        text_distances = cosine_distances(candidate_text_emb, interacted_text_emb).flatten()
        text_min = text_distances.min()
        
        return pd.Series({'text_min_dist': text_min})
    
    def compute_normalized_cooccurrence_stats(self, row):
        candidate_item = row['item_id']
        interacted_items = row['interacted_items']
        
        candidate_users = self.item_users_dict.get(candidate_item, set())
        candidate_popularity = len(candidate_users)
        
        cooccurrences = []
        for item in interacted_items:
            users = self.item_users_dict.get(item, set())
            co_users = candidate_users.intersection(users)
            co_count = len(co_users)
            normalized = co_count / candidate_popularity if candidate_popularity > 0 else 0
            cooccurrences.append(normalized)
        
        if cooccurrences:
            avg_co = np.mean(cooccurrences)
            max_co = np.max(cooccurrences)
            min_co = np.min(cooccurrences)
            median_co = np.median(cooccurrences)
            var_co = np.var(cooccurrences)
        
        return pd.Series({
            'avg_normalized_cooccurrence': avg_co,
            'max_normalized_cooccurrence': max_co,
            'min_normalized_cooccurrence': min_co,
            'median_normalized_cooccurrence': median_co,
            'var_normalized_cooccurrence': var_co
        })
    
    def compute_stats(self, dataset='val'):
        if dataset not in self.recommendations_paths:
            self.logger.error(f"Dataset '{dataset}' не определен в путях рекомендаций.")
            return
        
        path = self.recommendations_paths[dataset]
        self.logger.info(f"Обработка датасета '{dataset}' из файла {path}...")
        
        try:
            recommendations_df = pd.read_parquet(path)
            self.logger.info(f"Файл {path} загружен: {len(recommendations_df)} рекомендаций.")
            
            # Вычисление текстовых расстояний
            self.logger.info("Вычисление статистик текстовых расстояний...")
            distance_stats_df = recommendations_df.parallel_apply(self.compute_distance_stats, axis=1)
            recommendations_df = pd.concat([recommendations_df, distance_stats_df], axis=1)
            
            # Объединение с пользовательскими взаимодействиями
            self.logger.info("Объединение с пользовательскими взаимодействиями...")
            recommendations_df = recommendations_df.merge(self.user_items_df, on='user_id', how='left')
            recommendations_df['interacted_items'] = recommendations_df['interacted_items'].parallel_apply(

                lambda x: x if isinstance(x, set) else set()
            )
            
            # Вычисление статистик со-встречающихся взаимодействий
            self.logger.info("Вычисление статистик нормализованных со-встречающихся взаимодействий...")
            cooccurrence_stats_df = recommendations_df.parallel_apply(self.compute_normalized_cooccurrence_stats, axis=1)
            recommendations_df = pd.concat([recommendations_df, cooccurrence_stats_df], axis=1)
            
            # Удаление временного столбца
            recommendations_df = recommendations_df.drop(columns=['interacted_items'])
            
            # Сохранение обратно в Parquet
            recommendations_df.to_parquet(path)
            self.logger.info(f"Датасет '{dataset}' успешно обработан и сохранен в {path}.")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке датасета '{dataset}': {e}")
            raise

