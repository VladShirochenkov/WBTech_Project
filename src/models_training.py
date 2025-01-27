import os
import pandas as pd
import threadpoolctl
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel, ImplicitALSWrapperModel
from implicit.nearest_neighbours import TFIDFRecommender, BM25Recommender
from implicit.als import AlternatingLeastSquares
from src.config import Config
from src.logger import setup_logger

class ModelsTraining:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.num_threads = os.cpu_count()
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        threadpoolctl.threadpool_limits(1, "blas")
        self._load_data()

    def _load_data(self):
        try:
            data_path = self.config.data['processed_dir']
            self.first_train = pd.read_parquet(os.path.join(data_path, 'first_train_2.parquet'))
            self.first_val = pd.read_parquet(os.path.join(data_path, 'first_val_2.parquet'))
            self.warm_test = pd.read_parquet(os.path.join(data_path, 'warm_test_2.parquet'))
            # Если есть другие файлы, добавьте их здесь
            self.first_train_df = Dataset.construct(self.first_train)
            self.first_val_df = Dataset.construct(self.first_val)
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def train_itemknn_bm25(self):
        self.logger.info("Обучение модели ImplicitItemKNN с BM25Recommender")
        try:
            model = ImplicitItemKNNWrapperModel(
                model=BM25Recommender(
                    K=50, 
                    num_threads=self.num_threads, 
                    K1=0.8, 
                    B=0.3
                ),
                verbose=False
            )
            model.fit(self.first_train_df)
            return model
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели ItemKNN BM25: {e}")
            raise

    def train_itemknn_tfidf(self):
        self.logger.info("Обучение модели ImplicitItemKNN с TFIDFRecommender")
        try:
            model = ImplicitItemKNNWrapperModel(
                model=TFIDFRecommender(
                    K=50, 
                    num_threads=self.num_threads
                ),
                verbose=False
            )
            model.fit(self.first_train_df)
            return model
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели ItemKNN TFIDF: {e}")
            raise

    def train_als(self):
        self.logger.info("Обучение модели ImplicitALS")
        try:
            als = AlternatingLeastSquares(
                factors=300, 
                regularization=0.01, 
                alpha=150, 
                iterations=9, 
                random_state=0, 
                use_gpu=False,
                num_threads=self.num_threads
            )
            model = ImplicitALSWrapperModel(als)
            model.fit(self.first_train_df)
            return model
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели ALS: {e}")
            raise

    def generate_recommendations(self, model, users, dataset, k=50, filter_viewed=False, output_path=''):
        self.logger.info(f"Генерация рекомендаций для {len(users)} пользователей")
        try:
            recommendations = model.recommend(users, dataset, k=k, filter_viewed=filter_viewed)
            recommendations.to_parquet(output_path)
            self.logger.info(f"Рекомендации сохранены в {output_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при генерации рекомендаций: {e}")
            raise

    def train_and_predict(self):
        # Обучение моделей
        itemknn_bm25 = self.train_itemknn_bm25()
        itemknn_tfidf = self.train_itemknn_tfidf()
        als_model = self.train_als()

        # Генерация рекомендаций для валидации
        users_val = self.first_val[Columns.User].unique()
        processed_data_path = self.config.data['processed_dir']

        self.generate_recommendations(
            model=itemknn_bm25, 
            users=users_val, 
            dataset=self.first_train_df, 
            k=50, 
            filter_viewed=False, 
            output_path=os.path.join(processed_data_path, 'bm_25_recommendations_df_50_val.parquet')
        )
        self.generate_recommendations(
            model=itemknn_bm25, 
            users=self.warm_test[Columns.User].unique(), 
            dataset=self.first_train_df, 
            k=50, 
            filter_viewed=False, 
            output_path=os.path.join(processed_data_path, 'bm_25_recommendations_df_50_test.parquet')
        )
        self.generate_recommendations(
            model=itemknn_tfidf, 
            users=users_val, 
            dataset=self.first_train_df, 
            k=50, 
            filter_viewed=False, 
            output_path=os.path.join(processed_data_path, 'tfidf_recommendations_df_50_val.parquet')
        )
        self.generate_recommendations(
            model=itemknn_tfidf, 
            users=self.warm_test[Columns.User].unique(), 
            dataset=self.first_train_df, 
            k=50, 
            filter_viewed=False, 
            output_path=os.path.join(processed_data_path, 'tfidf_recommendations_df_50_test.parquet')
        )
        self.generate_recommendations(
            model=als_model, 
            users=users_val, 
            dataset=self.first_train_df, 
            k=50, 
            filter_viewed=False, 
            output_path=os.path.join(processed_data_path, 'ials_recommendations_df_50_val.parquet')
        )
        self.generate_recommendations(
            model=als_model, 
            users=self.warm_test[Columns.User].unique(), 
            dataset=self.first_train_df, 
            k=50, 
            filter_viewed=False, 
            output_path=os.path.join(processed_data_path, 'ials_recommendations_df_50_test.parquet')
        )
