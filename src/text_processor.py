import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
from .config import Config
from .logger import setup_logger

tqdm.pandas()

class TextProcessor:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.processed_dir = os.path.join(os.path.dirname(__file__), '..', self.config.data['processed_dir'])
        self.text_input = os.path.join(self.config.text['input_file'])
        self.text_output = os.path.join(self.processed_dir, 'text_wo_desc.parquet')
        self.columns_to_drop = [
            'Цвет', 'Длина юбки\платья', 'Длина упаковки', 'Высота упаковки', 'Ширина упаковки',
            'ИКПУ', 'Рост модели на фото', 'Код упаковки', 'Номер сертификата соответствия',
            'Дата регистрации сертификата/декларации', 'Ставка НДС', 'Номер декларации соответствия',
            'Дата окончания действия сертификата/декларации','Пол'
        ]
        self.columns_to_process = []  # Будет инициализироваться после загрузки данных

    def load_data(self):
        self.logger.info(f"Загрузка текстовых данных из {self.text_input}")
        try:
            self.text_df = pd.read_parquet(self.text_input)
            self.logger.info(f"Текстовые данные загружены успешно с формой: {self.text_df.shape}")
        except Exception as e:
            self.logger.error(f"Не удалось загрузить текстовые данные: {e}")
            raise

    def preprocess_text(self):
        self.logger.info("Начало предобработки текстовых данных")
        try:
            # Заполнение пропущенных значений
            self.text_df['colornames'] = self.text_df['colornames'].fillna(self.text_df['Цвет'])
            self.text_df['Длина юбки/платья'] = self.text_df['Длина юбки/платья'].fillna(self.text_df['Длина юбки\платья'])

            # Удаление ненужных столбцов
            self.text_df.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')

            self.columns_to_process = [col for col in self.text_df.columns if col not in ['item_id', 'description']]

            # Очистка столбца 'colornames'
            self.text_df['colornames'] = self.text_df['colornames'].progress_apply(self.clean_colornames)

            self.logger.info("Предобработка текстовых данных завершена")
        except Exception as e:
            self.logger.error(f"Ошибка предобработки текстовых данных: {e}")
            raise

    @staticmethod
    def clean_colornames(color_list):
        if not isinstance(color_list, (list, np.ndarray)):
            return []

        cleaned = []
        for color in color_list:
            if isinstance(color, str):
                color_cleaned = color.strip().strip("'\"").lower()
                if color_cleaned:
                    cleaned.append(color_cleaned)
        return cleaned

    @staticmethod
    def flatten_lists(value):

        if isinstance(value, (list, np.ndarray)):
            flat_list = []
            for item in value:
                flat_list.extend(TextProcessor.flatten_lists(item))
            return flat_list
        else:
            return [value]

    @staticmethod
    def is_value_na(value):
        if isinstance(value, (list, np.ndarray)):
            flat = TextProcessor.flatten_lists(value)
            return all(pd.isna(item) for item in flat)
        else:
            return pd.isna(value)

    def process_value(self, value):
        try:
            if self.is_value_na(value):
                return 'пропуск'
            if isinstance(value, (list, np.ndarray)):
                flat_elements = self.flatten_lists(value)
                flat_elements = [item for item in flat_elements if not pd.isna(item)]
                value = ' '.join(map(str, flat_elements))
            else:
                value = str(value)

            cleaned = re.findall(r'[^\W\d_]+|\d+|-', value, re.UNICODE)
            cleaned = ' '.join(cleaned)
            cleaned = cleaned.lower()

            if not cleaned.strip():
                return 'пропуск'

            return cleaned
        except Exception as e:
            self.logger.error(f"Ошибка обработки значения: {value} (тип: {type(value)})")
            raise e

    def create_combined_text(self, row):
        parts = []
        for col in self.columns_to_process:
            value = row[col]
            processed = self.process_value(value)
            part = f"{col}: {processed}"
            parts.append(part)
        return '\n'.join(parts)

    def generate_text_column(self):
        self.logger.info("Генерация комбинированного текстового столбца")
        try:
            self.text_df['text'] = self.text_df.progress_apply(self.create_combined_text, axis=1)
            self.logger.info("Комбинированный текстовый столбец успешно создан")
        except Exception as e:
            self.logger.error(f"Ошибка генерации текстового столбца: {e}")
            raise

    def save_data(self):
        self.logger.info(f"Сохранение обработанных текстовых данных в {self.text_output}")
        try:
            self.text_df[['item_id', 'text']].to_parquet(self.text_output, index=False)
            self.logger.info("Обработанные текстовые данные сохранены успешно")
        except Exception as e:
            self.logger.error(f"Не удалось сохранить обработанные текстовые данные: {e}")
            raise

    def process(self):
        self.load_data()
        self.preprocess_text()
        self.generate_text_column()
        self.save_data()
