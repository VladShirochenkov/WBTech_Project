import pandas as pd
import os
from .config import Config
from .logger import setup_logger
from pathlib import Path

class RecommendationProcessor:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.processed_dir = Path(self.config.data.get('processed_dir', 'data/processed/'))
        self.items_file = Path(self.config.items.get('file', 'data/processed/items_full.parquet'))
    
    def load_parquet(self, filepath):
        path = Path(filepath)
        self.logger.info(f"Загрузка файла: {path}")
        if not path.exists():
            self.logger.error(f"Файл {path} не найден.")
            raise FileNotFoundError(f"Файл {path} не найден.")
        return pd.read_parquet(path)
    
    def rename_columns(self, df, prefix):
        rename_map = {"score": f"{prefix}_score", "rank": f"{prefix}_rank"}
        self.logger.info(f"Переименование столбцов: {rename_map}")
        return df.rename(rename_map, axis=1)
    
    def process_recommendations(self, train_val='val'):
        if train_val not in ['val', 'test']:
            self.logger.error("Неверный тип диапазона данных. Используйте 'val' или 'test'.")
            raise ValueError("Неверный тип диапазона данных. Используйте 'val' или 'test'.")
        
        # Получение путей из конфигурации
        rec_config = self.config.recommendations.get(train_val, {})
        ial_recs_path = rec_config.get('ial_recommendations')
        bm25_recs_path = rec_config.get('bm25_recommendations')
        tfidf_recs_path = rec_config.get('tfidf_recommendations')
        text_recs_path = rec_config.get('text_recommendations')
        weight_input_path = rec_config.get('weight_input')
        output_path = rec_config.get('output')
        
        # Проверка наличия всех необходимых путей
        required_paths = [ial_recs_path, bm25_recs_path, tfidf_recs_path, text_recs_path, weight_input_path]
        if not all(required_paths):
            self.logger.error(f"Не все пути для '{train_val}' обработаны в конфигурации.")
            raise ValueError(f"Не все пути для '{train_val}' обработаны в конфигурации.")
        
        # Загрузка данных
        self.logger.info(f"Загрузка данных для '{train_val}'...")
        ial_recs = self.load_parquet(ial_recs_path)
        bm25_recs = self.load_parquet(bm25_recs_path)
        tfidf_recs = self.load_parquet(tfidf_recs_path)
        text_recs = self.load_parquet(text_recs_path)
        weight_df = self.load_parquet(weight_input_path)[['user_id', 'item_id', 'weight']].drop_duplicates()
        
        # Переименование столбцов
        self.logger.info(f"Переименование столбцов для '{train_val}'...")
        bm25_recs = self.rename_columns(bm25_recs, 'bm_25')
        tfidf_recs = self.rename_columns(tfidf_recs, 'tfidf')
        ial_recs = self.rename_columns(ial_recs, 'ials')
        text_recs = text_recs.rename({"rank": "text_rank"}, axis=1)
        
        # Объединение рекомендаций
        self.logger.info(f"Объединение рекомендаций для '{train_val}'...")
        recommendations_df = pd.merge(
            bm25_recs,
            ial_recs,
            how='outer',
            on=['user_id', 'item_id']
        )
        recommendations_df = pd.merge(
            recommendations_df,
            text_recs,
            how='outer',
            on=['user_id', 'item_id']
        )
        recommendations_df = pd.merge(
            recommendations_df,
            tfidf_recs,
            how='outer',
            on=['user_id', 'item_id']
        )
        
        # Объединение с весами
        self.logger.info(f"Объединение с весами для '{train_val}'...")
        recommendations_df = pd.merge(
            recommendations_df,
            weight_df,
            how='left',
            on=['user_id', 'item_id']
        )
        
        # Заполнение пропусков и изменение типов данных
        self.logger.info(f"Заполнение пропусков и изменение типов данных для '{train_val}'...")
        recommendations_df = recommendations_df.fillna({
            'bm_25_score': -101,
            'bm_25_rank': 101,
            'tfidf_score': -101,
            'tfidf_rank': 101,
            'ials_score': -101,
            'ials_rank': 101,
            'text_rank': 101,
            'weight': 0
        }).astype({
            'user_id': 'int32',
            'item_id': 'int32',
            'weight': 'float16',
            'bm_25_score': 'float16',
            'bm_25_rank': 'int16',
            'tfidf_score': 'float16',
            'tfidf_rank': 'int16',
            'ials_score': 'float16',
            'ials_rank': 'int16',
            'text_rank': 'int16'
        })
        
        # Объединение с информацией по товарам
        self.logger.info(f"Объединение с информацией по товарам для '{train_val}'...")
        items = self.load_parquet(self.items_file)
        recommendations_df = recommendations_df.merge(
            items[['item_id', 'item_popul']],
            on='item_id',
            how='left'
        )
        
        # Сохранение результирующего DataFrame
        self.logger.info(f"Сохранение рекомендаций в файл: {output_path}")
        recommendations_df.to_parquet(output_path)
        
        self.logger.info(f"Обработка рекомендаций для '{train_val}' завершена успешно.")
