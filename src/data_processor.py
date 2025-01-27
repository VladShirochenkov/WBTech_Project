import pandas as pd
import os
from .config import Config
from .logger import setup_logger
from rectools import Columns
from rectools.dataset import Dataset

class DataProcessor:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.processed_dir = os.path.join(os.path.dirname(__file__), '..', self.config.data['processed_dir'])
        self.processing_config = self.config.processing
    
    def load_data(self):
        self.logger.info("Загрузка датасетов")
        try:
            self.first_train = pd.read_parquet(os.path.join(self.processed_dir, 'first_train_2.parquet'))
            self.first_val = pd.read_parquet(os.path.join(self.processed_dir, 'first_val_2.parquet'))
            self.warm_test = pd.read_parquet(os.path.join(self.processed_dir, 'warm_test_2.parquet'))
            self.cold_test = pd.read_parquet(os.path.join(self.processed_dir, 'cold_test_2.parquet'))
            self.logger.info("Датасеты успешно загружены")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке датасетов: {e}")
            raise
    
    def process_data(self):
        self.logger.info("Начало обработки данных")
        self.load_data()
        try:
            # Переименование столбцов
            self.first_train = self.first_train.rename(
                columns={'dt': Columns.Datetime, 'nm_id': 'item_id', 'wbuser_id': 'user_id'}
            )
            
            for df in [self.first_val, self.warm_test, self.cold_test]:
                df['weight'] = 1
                df.rename(columns={'dt': Columns.Datetime, 'nm_id': 'item_id', 'wbuser_id': 'user_id'}, inplace=True)
            
            # Вычисление весов для обучающей выборки
            train_df_weights = self.first_train.groupby(["user_id", "item_id"]).agg({"item_id": "count",}).rename(columns={"item_id": "ui_inter",}).reset_index()
            total_users_interactions_count = train_df_weights[["user_id", "ui_inter"]].groupby("user_id").sum().rename(columns={"ui_inter": "u_total_inter"})
            train_df_weights = train_df_weights.join(total_users_interactions_count, on="user_id", how="left")
            train_df_weights["weight"] = train_df_weights["ui_inter"] / train_df_weights["u_total_inter"]
            
            # Объединение весов с обучающей выборкой
            self.first_train = self.first_train.merge(train_df_weights, on=["user_id", "item_id"], how="left")
            self.first_train["u_entry"] = self.first_train.groupby(["user_id"]).cumcount() + 1
            self.first_train["ui_entry"] = self.first_train.groupby(["user_id", "item_id"]).cumcount() + 1
            self.first_train["ui_entry_inter_ratio"] = self.first_train["ui_entry"] / self.first_train["ui_inter"]
            self.first_train["cum_weight"] = self.first_train["weight"] * self.first_train["ui_entry_inter_ratio"]
            self.first_train = self.first_train.drop(columns='weight')
            self.first_train = self.first_train.rename(columns={'cum_weight': 'weight'})
            
            # Сохранение обработанных данных
            self.first_train[['user_id', 'item_id', 'datetime', 'weight']].to_parquet(os.path.join(self.processed_dir, 'first_train_2.parquet'), index=False)
            self.first_val.to_parquet(os.path.join(self.processed_dir, 'first_val_2.parquet'), index=False)
            self.warm_test.to_parquet(os.path.join(self.processed_dir, 'warm_test_2.parquet'), index=False)
            self.cold_test.to_parquet(os.path.join(self.processed_dir, 'cold_test_2.parquet'), index=False)
            self.logger.info("Обработка данных успешно завершена")
        except Exception as e:
            self.logger.error(f"Ошибка во время обработки данных: {e}")
            raise
        