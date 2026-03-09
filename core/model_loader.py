
#loading pre-trained models (Autoencoder, KMeans, LightGBM, LabelEncoder)

import joblib
from pathlib import Path


class ModelLoader:
    def __init__(self, model_dir='model'):
        self.model_dir = Path(model_dir)
        self._ae_pipeline = None
        self._kmeans_pipeline = None
        self._lgbm_pipeline = None
        self._label_encoder = None
    
    @property
    def ae_pipeline(self):
        if self._ae_pipeline is None:
            path = self.model_dir / 'ae_pipeline.pkl'
            if not path.exists():
                raise FileNotFoundError(f"Autoencoder Model not found: {path}")
            self._ae_pipeline = joblib.load(path)
        return self._ae_pipeline
    
    @property
    def kmeans_pipeline(self):
        if self._kmeans_pipeline is None:
            path = self.model_dir / 'kmeans_pipeline.pkl'
            if not path.exists():
                raise FileNotFoundError(f"KMeans Model not found: {path}")
            self._kmeans_pipeline = joblib.load(path)
        return self._kmeans_pipeline
    
    @property
    def lgbm_pipeline(self):
        if self._lgbm_pipeline is None:
            path = self.model_dir / 'lgbm_pipeline.pkl'
            if not path.exists():
                raise FileNotFoundError(f"LightGBM Model not found: {path}")
            self._lgbm_pipeline = joblib.load(path)
        return self._lgbm_pipeline
    
    @property
    def label_encoder(self):
        if self._label_encoder is None:
            path = self.model_dir / 'label_encoder.pkl'
            if not path.exists():
                raise FileNotFoundError(f"LabelEncoder not found: {path}")
            self._label_encoder = joblib.load(path)
        return self._label_encoder
    
    def load_all(self):
        _ = self.ae_pipeline
        _ = self.kmeans_pipeline
        _ = self.lgbm_pipeline
        _ = self.label_encoder
        return self