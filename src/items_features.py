import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from src.logger import setup_logger
from src.config import Config

class ItemsFeatures:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.model = None

    def load_model(self):
        try:
            model_name = self.config.embedding.get('model_name', 'intfloat/multilingual-e5-small')
            self.logger.info(f"Загрузка модели SentenceTransformer: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.logger.info("Модель успешно загружена")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def load_items_text(self):
        try:
            input_path = os.path.join(os.path.dirname(__file__), '..', self.config.text['output_file'])
            self.logger.info(f"Загрузка текстовых данных из {input_path}")
            items = pd.read_parquet(input_path)
            self.logger.info(f"Текстовые данные успешно загружены: {items.shape[0]} записей")
            return items
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке текстовых данных: {e}")
            raise

    def generate_text_embeddings(self, items):
        try:
            self.logger.info("Генерация эмбеддингов текстов")
            item_texts = items['text'].tolist()
            item_embeddings = self.model.encode(
                item_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self.logger.info("Эмбеддинги успешно сгенерированы")
            return item_embeddings
        except Exception as e:
            self.logger.error(f"Ошибка при генерации эмбеддингов: {e}")
            raise

    def compute_item_popul(self, train_input_path):
        try:
            self.logger.info(f"Загрузка тренировочных данных из {train_input_path}")
            train_df = pd.read_parquet(train_input_path)

            self.logger.info("Вычисление количества взаимодействий (interactions) по user_id и item_id")
            train_df_weights = (
                train_df.groupby(["user_id", "item_id"])
                .agg(ui_inter=("item_id", "count"))
                .reset_index()
            )

            self.logger.info("Агрегация взаимодействий по item_id")
            item_interactions = (
                train_df_weights.groupby("item_id")
                .agg(item_count=("ui_inter", "sum"))
            )

            self.logger.info("Вычисление популярности айтемов")
            item_interactions["item_popul"] = item_interactions["item_count"] / item_interactions["item_count"].sum()
            
            self.logger.info("Популярность айтемов успешно вычислена")
            return item_interactions[['item_popul']]
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении популярности айтемов: {e}")
            raise

    def create_feature_dataframe(self, items, embeddings, item_popul):
        try:
            self.logger.info("Создание DataFrame с признаками для айтемов")

            # Удаляем столбец 'text' и объединяем эмбеддинги
            items = items.drop(columns=['text'])
            embedding_dim = embeddings.shape[1]
            
            embedding_cols = [f'text_{i}' for i in range(embedding_dim)]
            embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols, index=items.index)
            items = pd.concat([items, embeddings_df], axis=1)

            # Объединяем с популярностью айтемов
            items = items.join(item_popul, on="item_id", how="left")

            self.logger.info("DataFrame с признаками успешно создан")
            return items
        except Exception as e:
            self.logger.error(f"Ошибка при создании DataFrame с признаками: {e}")
            raise

    def save_features(self, features_df):
        try:
            output_file = self.config.embedding.get('output_file', 'items_full.parquet')
            output_path = os.path.join(os.path.dirname(__file__), '..', self.config.data['processed_dir'], output_file)
            self.logger.info(f"Сохранение признаков айтемов в {output_path}")
            features_df.to_parquet(output_path)
            self.logger.info("Признаки айтемов успешно сохранены")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении признаков: {e}")
            raise

    def process(self):
        try:
            self.load_model()
            items_text = self.load_items_text()
            embeddings = self.generate_text_embeddings(items_text)

            train_input_path = os.path.join(os.path.dirname(__file__), '..', self.config.processing['train_input'])
            item_popul = self.compute_item_popul(train_input_path)

            features_df = self.create_feature_dataframe(items_text, embeddings, item_popul)
            self.save_features(features_df)

        except Exception as e:
            self.logger.error(f"Ошибка при обработке признаков айтемов: {e}")
            raise
