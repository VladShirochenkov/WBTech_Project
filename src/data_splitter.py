import pandas as pd
import os
from .config import Config
from .logger import setup_logger

class DataSplitter:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.raw_path = os.path.join(os.path.dirname(__file__), '..', self.config.data['raw_path'])
        self.processed_dir = os.path.join(os.path.dirname(__file__), '..', self.config.data['processed_dir'])
        self.random_seed = self.config.split['random_seed']
        self.fraction = self.config.split['fraction']
        self.split_frac = self.config.split['split_frac']
    
    def load_data(self):
        self.logger.info(f"Загрузка данных из {self.raw_path}")
        try:
            self.df = pd.read_parquet(self.raw_path)       
            self.logger.info(f"Данные загружены с размером: {self.df.shape}")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}")
            raise
    
    def split_users(self, df, user_column='wbuser_id', frac=0.5, random_state=None):
        unique_users = df[user_column].unique()
        unique_users_df = pd.DataFrame(unique_users, columns=[user_column])
        shuffled_users = unique_users_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_index = int(len(shuffled_users) * frac)
        first_part_users = shuffled_users.iloc[:split_index][user_column].tolist()
        second_part_users = shuffled_users.iloc[split_index:][user_column].tolist()
        return first_part_users, second_part_users

    def save_data(self, df, filename):
        save_path = os.path.join(self.processed_dir, filename)
        try:
            df.to_parquet(save_path, index=False)
            self.logger.info(f"Сохранили {filename} в {save_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения {filename}: {e}")
            raise

    def split_data(self):
        self.logger.info("Начало разделения данных")
        self.load_data()
        df = self.df.copy()
    
        # Фильтрация пользователей
        only_day_11_users = df.groupby('wbuser_id')['day'].unique()
        only_day_11_users = [user for user, days in only_day_11_users.items() if set(days) == {11}]
        
        both_days_users = df.groupby('wbuser_id')['day'].unique()
        both_days_users = [user for user, days in both_days_users.items() if set(days) == {10, 11}]
        
        count_only_day_11_users = len(only_day_11_users)
        count_both_days_users = len(both_days_users)
        self.logger.info(f"Only day 11 users: {count_only_day_11_users}, Both days users: {count_both_days_users}")
        
        # Выбор холодных пользователей
        only_day_11_series = pd.Series(only_day_11_users)
        cold_users_selected = only_day_11_series.sample(frac=self.fraction, random_state=self.random_seed).tolist()
        cold_users = df[df['wbuser_id'].isin(cold_users_selected)].copy()
        df = df[~df['wbuser_id'].isin(cold_users_selected)].copy()
        
        # Оставшиеся только день 11 пользователи
        remaining_only_day_11_users = list(set(only_day_11_users) - set(cold_users_selected))
        only_day_11_remaining_df = df[df['wbuser_id'].isin(remaining_only_day_11_users)].copy()
        only_day_11_remaining_df = only_day_11_remaining_df.sort_values(['wbuser_id', 'dt'])
        
        # Теплые пользователи
        warm_users = only_day_11_remaining_df.groupby('wbuser_id').tail(4).copy()
        warm_user_indexes = warm_users.index
        df = df.drop(warm_user_indexes).copy()
        
        # Дополнительное разделение теплых пользователей
        first_val_warm_users, warm_test_users = self.split_users(warm_users, user_column='wbuser_id',
                                                                  frac=self.split_frac, random_state=self.random_seed)
        first_val = warm_users[warm_users['wbuser_id'].isin(first_val_warm_users)].copy()
        warm_test = warm_users[warm_users['wbuser_id'].isin(warm_test_users)].copy()
        
        # Подготовка тестовых и валидационных наборов
        cold_test = cold_users.copy()
        
        # Удаление ненужных столбцов
        for dataframe in [df, warm_test, cold_test, first_val]:
            dataframe.drop(['day', 'hour'], axis=1, inplace=True, errors='ignore')
            dataframe['wbuser_id'] = dataframe['wbuser_id'].astype('int64')
            dataframe['nm_id'] = dataframe['nm_id'].astype('int64')
            
        # Сохранение данных
        self.save_data(df, 'first_train_2.parquet')
        self.save_data(cold_test, 'cold_test_2.parquet')
        self.save_data(warm_test, 'warm_test_2.parquet')
        self.save_data(first_val, 'first_val_2.parquet')
        self.logger.info("Разделенные данные успешно сохранены")
    

