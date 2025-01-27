import os
import logging
import pandas as pd
import numpy as np
from rectools.dataset import Dataset
import threadpoolctl
from implicit.als import AlternatingLeastSquares
from rectools.models import ImplicitALSWrapperModel
import faiss
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .config import Config

class InnerProduct:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger('WBTech_Project')
        self.logger.info("Инициализация InnerProduct")
        self.setup_environment()
        self.load_data()
        self.train_model()
        self.create_embeddings()
        self.build_faiss_index()

    def setup_environment(self):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        threadpoolctl.threadpool_limits(1, "blas")
        self.num_threads = os.cpu_count()
        self.logger.info(f"Окружение настроено. Число потоков: {self.num_threads}")

    def load_data(self):
        try:
            self.logger.info(f"Загрузка данных из {self.config.inner_product['first_train_path']}")
            self.first_train = pd.read_parquet(self.config.inner_product['first_train_path'])
            self.recommendations_val = pd.read_parquet(self.config.inner_product['recommendations_val_path'])
            self.recommendations_test = pd.read_parquet(self.config.inner_product['recommendations_test_path'])
            self.logger.info("Данные загружены успешно.")
        except Exception as e:
            self.logger.exception("Ошибка при загрузке данных.")
            raise e

    def train_model(self):
        try:
            self.logger.info("Начало обучения модели Implicit ALS.")
            first_train_df = Dataset.construct(self.first_train)
            self.ials_model = ImplicitALSWrapperModel(
                AlternatingLeastSquares(
                    factors=self.config.inner_product['model_parameters']['factors'],
                    regularization=self.config.inner_product['model_parameters']['regularization'],
                    alpha=self.config.inner_product['model_parameters']['alpha'],
                    iterations=self.config.inner_product['model_parameters']['iterations'],
                    random_state=self.config.inner_product['model_parameters']['random_state'],
                    use_gpu=self.config.inner_product['model_parameters']['use_gpu'],
                    num_threads=self.num_threads
                ),
            )
            self.ials_model.fit(first_train_df)
            self.logger.info("Модель обучена успешно.")
        except Exception as e:
            self.logger.exception("Ошибка при обучении модели.")
            raise e

    def create_embeddings(self):
        try:
            self.logger.info("Создание эмбеддингов пользователей и товаров.")
            user_ids = self.first_train['user_id'].unique()
            item_ids = self.first_train['item_id'].unique()
            
            users, items = self.ials_model.get_vectors()
            
            user_columns = [f'user_{i}' for i in range(users.shape[1])]
            self.users_emb = pd.DataFrame(users, index=user_ids, columns=user_columns)
            self.users_emb.to_parquet(self.config.inner_product['output_users_path'])
            
            item_columns = [f'item_{i}' for i in range(items.shape[1])]
            self.items_emb = pd.DataFrame(items, index=item_ids, columns=item_columns)
            
            self.logger.info("Эмбеддинги созданы успешно.")
        except Exception as e:
            self.logger.exception("Ошибка при создании эмбеддингов.")
            raise e

    def build_faiss_index(self):
        try:
            self.logger.info("Создание FAISS индекса.")
            self.users_emb_normalized = normalize(self.users_emb.values, axis=1)
            self.items_emb_normalized = normalize(self.items_emb.values, axis=1)
            
            d = self.users_emb_normalized.shape[1] 
            self.index = faiss.IndexFlatIP(d)  
            self.index.add(self.items_emb_normalized)  
            self.logger.info("FAISS индекс создан и заполнен успешно.")
            
            # Создание словарей для быстрой индексации
            self.user_ids = self.users_emb.index.tolist()
            self.item_ids = self.items_emb.index.tolist()
            
            self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
            self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        except Exception as e:
            self.logger.exception("Ошибка при создании FAISS индекса.")
            raise e

    def get_inner_product(self, user_id, item_id):
        user_idx = self.user_id_to_idx.get(user_id)
        item_idx = self.item_id_to_idx.get(item_id)
        
        if user_idx is None or item_idx is None:
            self.logger.warning(f"Отсутствие индекса для user_id={user_id} или item_id={item_id}.")
            return np.nan
        
        user_vector = self.users_emb_normalized[user_idx]
        item_vector = self.items_emb_normalized[item_idx]
        return float(np.dot(user_vector, item_vector))

    def compute_inner_products(self):
        try:
            self.logger.info("Вычисление внутреннего произведения для validation набора.")
            self.recommendations_val['inner_product'] = self.recommendations_val.progress_apply(
                lambda row: self.get_inner_product(row['user_id'], row['item_id']), axis=1
            )
            self.recommendations_val.to_parquet(self.config.inner_product['output_val_path'])
            self.logger.info("Валидационный набор обновлён и сохранён.")
            
            self.logger.info("Вычисление внутреннего произведения для test набора.")
            self.recommendations_test['inner_product'] = self.recommendations_test.progress_apply(
                lambda row: self.get_inner_product(row['user_id'], row['item_id']), axis=1
            )
            self.recommendations_test.to_parquet(self.config.inner_product['output_test_path'])
            self.logger.info("Тестовый набор обновлён и сохранён.")
        except Exception as e:
            self.logger.exception("Ошибка при вычислении внутреннего произведения.")
            raise e
