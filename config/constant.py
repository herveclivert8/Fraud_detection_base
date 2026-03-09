
# Colors for fraud types
fraud_color_map = {
    'money laundering': '#E63946',
    'fraud by card': '#F4A261',
    'fraud by mule account': '#9B2226',
    'legitimate': '#2A9D8F'
}

# threshold for detection 
anomaly_threshold_percentile = 99 #percentile for anomaly threshold
confidence_threshold = 0.05  # confidence threshold for signals_frauds 

# Fraud types
fraud_types = ['money laundering', 'fraud by card', 'fraud by mule account', 'legitimate']

# columns to exclude from display
columns_to_exclude = ['Anomaly_Score', 'isAnomaly', 'Cluster', 'Pseudo_Labels', 'Prediction']