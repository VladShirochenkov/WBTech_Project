import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import PopularModel

from config import Config
from logger import setup_logger

class ColdRecos:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.processing_config = self.config.processing
        self.logger.info("Инициализация класса ColdRecos завершена.")

    def run(self):
        try:
            cold_test_path = self.processing_config.get('cold_test_output')
            if not cold_test_path:
                self.logger.error("Путь к 'cold_test_output' не найден в конфигурации.")
                return

            self.logger.info(f"Чтение файла cold_test: {cold_test_path}")
            cold_test = pd.read_parquet(cold_test_path)
            self.logger.info(f"Файл cold_test прочитан успешно. Количество записей: {len(cold_test)}")

            first_train_path = self.processing_config.get('first_train_output')
            self.logger.info(f"Чтение файла first_train: {first_train_path}")
            first_train = pd.read_parquet(first_train_path)
            self.logger.info(f"Файл first_train прочитан успешно. Количество записей: {len(first_train)}")

            self.logger.info("Создание объекта Dataset для first_train.")
            first_train_df = Dataset.construct(first_train)
            
            self.logger.info("Инициализация и обучение модели PopularModel.")
            pop_model = PopularModel()
            pop_model.fit(first_train_df)
            self.logger.info("Модель PopularModel обучена успешно.")
            
            self.logger.info("Генерация рекомендаций.")
            popular_reco_df = pop_model.recommend(
                users=cold_test[Columns.USER].unique(),
                dataset=first_train_df,
                k=10,
                filter_viewed=False,
            )
            self.logger.info(f"Рекомендации сгенерированы. Количество рекомендаций: {popular_reco_df.shape[0]}")
            
            # Сохранение рекомендаций в Parquet файл
            output_path = self.processing_config.get('cold_recos_output', 'cold_recos.parquet')
            self.logger.info(f"Сохранение рекомендаций в файл: {output_path}")
            popular_reco_df.to_parquet(output_path)
            self.logger.info("Рекомендации сохранены успешно.")
        
        except Exception as e:
            self.logger.exception(f"Произошла ошибка при выполнении ColdRecos: {e}")