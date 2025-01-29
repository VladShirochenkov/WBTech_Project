from src.data_splitter import DataSplitter
from src.data_processor import DataProcessor
from src.text_processor import TextProcessor
from src.items_features import ItemsFeatures 
from src.text_recommendations import RecommendationsText
from src.models_training import ModelsTraining
from src.recommendation_processor import RecommendationProcessor
from src.inner_product import InnerProduct
from src.compute_recommendation_stats import RecommendationStats
from src.recommendation_cluster import RecommendationCluster
from src.lgbm_train import LGBMTrain
from src.cold_recos import ColdRecos
from src.logger import setup_logger

def main():
    logger = setup_logger()
    logger.info("Запуск WBTech_Project")

    # Разделение данных
    splitter = DataSplitter()
    splitter.split_data()
        
    # Обработка данных
    processor = DataProcessor()
    processor.process_data()
        
    # Обработка текстовых данных
    text_processor = TextProcessor()
    text_processor.process()
        
    # Генерация и обработка признаков айтемов
    items_features = ItemsFeatures()
    items_features.process()
        
    recommender = RecommendationsText()
    recommender.recos()

    trainer = ModelsTraining()
    trainer.train_and_predict()

    # Обработка рекомендаций для валидации
    recommendation_processor = RecommendationProcessor()
    recommendation_processor.process_recommendations(train_val='val')
        
    # Обработка рекомендаций для тестирования
    recommendation_processor.process_recommendations(train_val='test')

    inner_product = InnerProduct()
    inner_product.compute_inner_products()

    # Инициализация класса RecommendationStats
    recommendation_stats = RecommendationStats()
    
    # Обработка val датасета
    recommendation_stats.compute_stats(dataset='val')
    
    # Обработка test датасета
    recommendation_stats.compute_stats(dataset='test')
        
        
    recommendation_cluster = RecommendationCluster()

    recommendation_cluster.compute_stats(dataset='val')

    recommendation_cluster.compute_stats(dataset='test')


    lgbm_trainer = LGBMTrain()

    lgbm_trainer.train()


    cold_recos = ColdRecos()

    cold_recos.run()

    logger.info("WBTech_Project успешно завершен")

if __name__ == "__main__":
    main()
