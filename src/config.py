import yaml
import os

class Config:
    def __init__(self, config_file='config.yaml'):
        config_path = os.path.join(os.path.dirname(__file__), '..', config_file)
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.data = cfg.get('data', {})
        self.logging = cfg.get('logging', {})
        self.split = cfg.get('split', {})
        self.processing = cfg.get('processing', {})
        self.text = cfg.get('text', {})
        self.embedding = cfg.get('embedding', {})  
        self.models = cfg.get('models', {})
        self.recommendations = cfg.get('recommendations', {})
        self.items = cfg.get('items', {})
        self.inner_product = cfg.get('inner_product', {})
        self.data_recos_stats = cfg.get('data_recos_stats', {})
        self.cluster = cfg.get('cluster', {})
        self.lgbm_train = cfg.get('lgbm_train', {})