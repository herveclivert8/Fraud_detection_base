
#Full pipeline : from the raw data to the final prediction
#  Steps:
#      anomalies detection (Autoencoder)
#      anomalies clustering (KMeans)
#      final prediction (LightGBM)

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing import preprocessing, preprocessing_lgbm
from features import rfm_features, signals_frauds
from config.constant import anomaly_threshold_percentile 


class FraudDetectionPipeline:
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.ae_pipeline = model_loader.ae_pipeline
        self.kmeans_pipeline = model_loader.kmeans_pipeline
        self.lgbm_pipeline = model_loader.lgbm_pipeline
        self.label_encoder = model_loader.label_encoder
    
    def preprocess_data(self, df):
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Convert integers into objects (categorials)
        for col in df.select_dtypes(include='int'):
            df[col] = df[col].astype('object')
        
        return df
    
    def detect_anomalies(self, df):
    
        X_AE, num_cols, cat_cols = preprocessing(df)
        X_AE['Anomaly_Score'] = self.ae_pipeline.score_samples(X_AE)
        threshold = np.percentile(X_AE['Anomaly_Score'], anomaly_threshold_percentile)
        X_AE['isAnomaly'] = self.ae_pipeline.predict(X_AE, threshold=threshold)
        df_result = df.copy()
        df_result.loc[X_AE.index, 'Anomaly_Score'] = X_AE['Anomaly_Score']
        df_result.loc[X_AE.index, 'isAnomaly'] = X_AE['isAnomaly']
        
        return df_result
    
    def cluster_anomalies(self, df):
        # Filter anomalies
        anomalies = df[df['isAnomaly'] == 1].copy()
        
        if anomalies.empty:
            # no anomaly detected
            df['Cluster'] = np.nan
            return df
        
        # Clustering
        X_KMEANS, num_cols, cat_cols = preprocessing(anomalies)
        X_KMEANS['Cluster'] = self.kmeans_pipeline.predict(X_KMEANS)
        
        # Adding to DataFrame
        df_result = df.copy()
        df_result.loc[X_KMEANS.index, 'Cluster'] = X_KMEANS['Cluster']
        
        return df_result
    
    def assign_pseudo_labels(self, df):
        df_cluster = df[df['Cluster'].notna()].copy()
        
        if df_cluster.empty:
            df['Pseudo_Labels'] = 'legitimate'
            return df
        
        # RFM Features and signals
        RFM = rfm_features(df_cluster)
        df_frauds, mapping = signals_frauds(RFM)
        
        # Adding pseudo-labels
        df_result = df.copy()
        df_result.loc[df_frauds.index, 'Pseudo_Labels'] = df_frauds['Pseudo_Labels']
        df_result['Pseudo_Labels'] = df_result['Pseudo_Labels'].fillna('legitimate')
        
        return df_result
    
    def predict_fraud_types(self, df):

        df_labels = df[df['Pseudo_Labels'].notna()].copy()
        
        if df_labels.empty:
            df['Prediction'] = 'legitimate'
            return df
        
        # Preprocessing for LightGBM
        X, y, cat_cols = preprocessing_lgbm(df_labels)
        
        # Prediction
        y_pred = self.lgbm_pipeline.predict(X)
        y_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Adding to DataFrame
        df_result = df.copy()
        df_result.loc[df_labels.index, 'Prediction'] = y_labels
        df_result['Prediction'] = df_result['Prediction'].fillna('legitimate')
        
        return df_result
    
    def run_full_pipeline(self, df):
        
        # Preparation
        df = self.preprocess_data(df)
        
        # Sequential steps
        df = self.detect_anomalies(df)
        df = self.cluster_anomalies(df)
        df = self.assign_pseudo_labels(df)
        df = self.predict_fraud_types(df)
        
        return df
    
    def predict_proba(self, df):
        X, y, cat_cols = preprocessing_lgbm(df)
        y_proba = self.lgbm_pipeline.predict_proba(X)
        return y_proba
    
    def get_class_names(self):
        classes_encoded = self.lgbm_pipeline.classes_
        classes = self.label_encoder.inverse_transform(classes_encoded)
        return classes