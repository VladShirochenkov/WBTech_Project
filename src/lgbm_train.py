import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRanker
from .config import Config
from .logger import setup_logger

class LGBMTrain:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.load_params()
    
    def load_params(self):
        # Загрузка параметров модели из конфигурации
        model_cfg = self.config.models.get('lgbm_ranker', {})
        self.best_params = {
            'learning_rate': model_cfg.get('learning_rate'),
            'n_estimators': model_cfg.get('n_estimators'),
            'max_depth': model_cfg.get('max_depth'),
            'num_leaves': model_cfg.get('num_leaves'),
            'min_child_samples': model_cfg.get('min_child_samples'),
            'reg_lambda': model_cfg.get('reg_lambda'),
            'colsample_bytree': model_cfg.get('colsample_bytree'),
            'subsample': model_cfg.get('subsample'),
            'objective': model_cfg.get('objective', 'lambdarank'),
            'metric': model_cfg.get('metric', 'map'),
            'verbosity': model_cfg.get('verbosity', -1),
            'random_state': model_cfg.get('random_state')
        }
        self.early_stopping_rounds = model_cfg.get('early_stopping_rounds')
    
    def get_group(self, df):
        group = df.groupby('user_id')['item_id'].count().values
        return np.array(group)
    
    def train(self):
        try:
            self.logger.info("Загрузка тренировочных и валидационных данных.")
            processed_dir = self.config.data.get('processed_dir', 'data/processed/')
            recommendations_val_path = os.path.join(
                processed_dir,
                self.config.cluster.get('recommendations', {}).get('val_file', 'recommendations_val.parquet')
            )
            recommendations_test_path = os.path.join(
                processed_dir,
                self.config.cluster.get('recommendations', {}).get('test_file', 'recommendations_test.parquet')
            )
            
            recommendations_df = pd.read_parquet(recommendations_val_path)
            
            unique_users = recommendations_df['user_id'].unique()
            train_users, val_users = train_test_split(
                unique_users, test_size=0.2, random_state=self.config.cluster.get('random_state', 42)
            )
            train_df = recommendations_df[recommendations_df['user_id'].isin(train_users)]
            val_df = recommendations_df[recommendations_df['user_id'].isin(val_users)]
            
            self.logger.info("Подготовка колонок признаков.")
            exclude_cols = ['user_id', 'item_id', 'weight', 'var_normalized_cooccurrence']
            cols = [col for col in recommendations_df.columns if col not in exclude_cols]
            
            self.logger.info("Подготовка групп для обучения.")
            group_train = self.get_group(train_df)
            group_val = self.get_group(val_df)
            
            self.logger.info("Инициализация LGBMRanker с лучшими параметрами.")
            listwise_model = LGBMRanker(**self.best_params, early_stopping_rounds=self.early_stopping_rounds)
            
            self.logger.info("Начало обучения модели.")
            listwise_model.fit(
                train_df[cols],
                train_df['weight'],
                group=group_train,
                eval_set=[(val_df[cols], val_df['weight'])],
                eval_group=[group_val],
                eval_metric='map',
                eval_at=(1, 5, 10),
                feature_name=cols,
            )
            self.logger.info("Обучение модели завершено.")
            
            self.logger.info("Загрузка тестовых данных для предсказаний.")
            test_df = pd.read_parquet(recommendations_test_path)
            
            self.logger.info("Генерация предсказаний на тестовых данных.")
            listwise_df = test_df[["user_id", "item_id"]].copy()
            listwise_df["listwise_score"] = listwise_model.predict(test_df[cols])
            
            self.logger.info("Сортировка и ранжирование предсказаний.")
            listwise_df = listwise_df.sort_values(["user_id", "listwise_score"], ascending=[True, False])

            listwise_df['rank'] = listwise_df.groupby('user_id').cumcount() + 1
            
            output_path = os.path.join(
                processed_dir,
                'listwise_df_text_final.parquet'
            )
            self.logger.info(f"Сохранение ранжированных предсказаний в {output_path}.")
            listwise_df.to_parquet(output_path)
            self.logger.info("Ранжированные предсказания успешно сохранены.")
        
        except Exception as e:
            self.logger.exception("Произошла ошибка во время обучения:")
            raise e
