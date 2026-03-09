
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix


class MetricsCalculator:
    
    def __init__(self, label_encoder):
        
        # LabelEncoder to convert the labels
        self.label_encoder = label_encoder
    
    # Calculate macro and weighted AUC scores.
    def compute_auc_scores(self, y_true, y_proba):
        try:
            auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            return {
                'auc_macro': auc_macro,
                'auc_weighted': auc_weighted
            }
        except Exception as e:
            print(f" AUC computation error : {e}")
            return {'auc_macro': 0.0, 'auc_weighted': 0.0}
    
    #compute ROC curves for each class
    def compute_roc_curves(self, y_true, y_proba):
        roc_data = {}
        n_classes = len(self.label_encoder.classes_)
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
            roc_data[i] = {
                'fpr': fpr,
                'tpr': tpr,
                'label': self.label_encoder.inverse_transform([i])[0]
            }
        
        return roc_data
    
    #compute classification report and confusion matrix 
    def compute_classification_metrics(self, y_true, y_pred):
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'report': report,
            'confusion_matrix': cm
        }
    
    def get_class_names(self):
        return self.label_encoder.classes_