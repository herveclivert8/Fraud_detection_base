import numpy as np
import pandas as pd

#Function to create RFM variables (Recency, Frequency, Monetary) and other useful signals to detect frauds after clustering
def rfm_features(df):
    df = df.copy()

    # ------ RECENCY (R) -------

    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0 = Monday, 6 = Sunday

    #Recency of transmitter account : number of days since last transaction
    last_tx = df.groupby('From Account')['Timestamp'].transform('max')
    df['Recency_Days'] = (df['Timestamp'].max() - last_tx).dt.days

    df['isNight'] = df['Hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)
    df['isWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # ------- FREQUENCY (F) ---------

    # Number of transaction per transmitter account 
    df['Freq_Tx'] = df.groupby('From Account')['Timestamp'].transform('count')

    # Number of unique 
    # Nombre de destinataires uniques par compte émetteur
    df['Unique_To_per_From'] = df.groupby('From Account')['To Account'].transform('nunique')

    # Burst : number of transactions within a short time interval (< 3min)
    Time_diff_Min = (
        df.sort_values(['From Account', 'Timestamp'])
        .groupby('From Account')['Timestamp']
        .diff()
        .dt.total_seconds() / 60
    )
    df['isBrust'] = Time_diff_Min.apply(lambda x: int(x <= 3 if pd.notnull(x) else 0))

    # ------ MONETARY (M) ------

    # Difference between paid amount / received
    df['Amount_Diff'] = df['Amount Paid'] - df['Amount Received']

    # Average and maximum amounts sent per account
    df['Amount_Mean'] = df.groupby('From Account')['Amount Paid'].transform('mean')
    df['Amount_Max'] = df.groupby('From Account')['Amount Paid'].transform('max')

    # Smurfing : frequent small amounts
    df['Small_Amount'] = (df['Amount Paid'] < 200).astype(int)
    df['Nb_Small_Tx'] = df.groupby('From Account')['Small_Amount'].transform('sum')

    # Log transformation for distribution stabilization
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Log_Amount_Diff'] = np.log1p(np.abs(df['Amount_Diff']))
    df['Log_Amount_Mean'] = np.log1p(df['Amount_Mean'])
    df['Log_Amount_Max'] = np.log1p(df['Amount_Max'])

    # ------- Others -------

    # Intra-bank transfert
    df['Same_Bank_Transfer'] = (df['From Bank'] == df['To Bank']).astype(int)

    return df


def signals_frauds(df, cluster_col='Cluster', confidence_threshold=0.05):
    df = df.copy()

    # --- Calcul des signaux ---
    df['High_Amount'] = (
        (df['Log_Amount_Mean'] > df['Log_Amount_Mean'].quantile(0.95)) |
        (df['Log_Amount_Max'] > df['Log_Amount_Max'].quantile(0.95))
    ).astype(int)
    df['isInternational'] = (df['Receiving Currency'] != df['Payment Currency']).astype(int)
    df['Freq_Small_Tx'] = (df['Nb_Small_Tx'] > df['Nb_Small_Tx'].quantile(0.9)).astype(int)
    df['Many_Dests'] = (df['Unique_To_per_From'] > df['Unique_To_per_From'].quantile(0.95)).astype(int)
    df['High_Freq_From'] = (df['Freq_Tx'] > df['Freq_Tx'].quantile(0.95)).astype(int)
    df['Similary'] = (
        (df['Receiving Currency'] == df['Payment Currency']) |
        (df.get('Same_Bank_Transfer', 0) == 1)
    ).astype(int)
    df['Wire_ACH_Bitcoin_Format'] = df['Payment Format'].isin(['Wire', 'Bitcoin', 'ACH']).astype(int)
    df['Very_Recent'] = (df['Recency_Days'] < 1).astype(int)
    df['Reactivation_Suspect'] = ((df['Recency_Days'] > 10) & (df['Freq_Tx'] > 2)).astype(int)
    df['Cash_Bitcoin_Format'] = df['Payment Format'].isin(['Cash', 'Bitcoin']).astype(int)
    df['Credit_Format'] = (df['Payment Format'] == 'Credit Card').astype(int)

    # Fraud types definition and their signals
    frauds = {
        'money laundering': ['High_Amount', 'isInternational', 'Freq_Small_Tx', 'Many_Dests',
                        'High_Freq_From', 'Similary', 'Wire_ACH_Bitcoin_Format'],
        'fraud by card': ['Very_Recent', 'isNight', 'isWeekend', 'isBrust', 'Credit_Format'],
        'fraud by mule account': ['Very_Recent', 'Reactivation_Suspect', 'Many_Dests',
                                   'High_Freq_From', 'Cash_Bitcoin_Format']
    }

    all_signals = list(set(sig for sigs in frauds.values() for sig in sigs))
    existing_signals = [s for s in all_signals if s in df.columns]
    cluster_profiles = df.groupby(cluster_col)[existing_signals].mean()

    cluster_scores = []

    # calculating cluster scores by payment type distribution 
    for cluster_id, row in cluster_profiles.iterrows():
        formats_present = set(df[df[cluster_col] == cluster_id]['Payment Format'].unique())

        possible_types = set()
        if any(f in formats_present for f in ['Wire', 'ACH', 'Bitcoin']):
            possible_types.add('money laundering')
        if 'Credit Card' in formats_present:
            possible_types.add('fraud by card')
        if any(f in formats_present for f in ['Cash', 'Bitcoin']):
            possible_types.add('fraud by mule account')

        for fraud_type in possible_types:
            signals = frauds[fraud_type]
            present_signals = [s for s in signals if s in row.index]
            if present_signals:
                score = row[present_signals].mean()
                cluster_scores.append((cluster_id, fraud_type, score))

    # Sort by score descending 
    cluster_scores.sort(key=lambda x: x[2], reverse=True)

    cluster_to_fraud = {}
    assigned_types = set()
    assigned_clusters = set()

    # Single assignment : one type per cluster, using the top score
    for cluster_id, fraud_type, score in cluster_scores:
        if (fraud_type not in assigned_types
                and cluster_id not in assigned_clusters
                and score >= confidence_threshold):
            cluster_to_fraud[cluster_id] = fraud_type
            assigned_types.add(fraud_type)
            assigned_clusters.add(cluster_id)

    # Unassigned clusters : remaining or legitimate
    remaining_types = set(frauds.keys()) - assigned_types

    for cluster_id in cluster_profiles.index:
        if cluster_id not in assigned_clusters:
            scores_for_cluster = [score for cid, _, score in cluster_scores if cid == cluster_id]
            max_score = max(scores_for_cluster) if scores_for_cluster else 0

            if max_score < confidence_threshold or not remaining_types:
                cluster_to_fraud[cluster_id] = "legitimate"
            else:
                fraud_type = remaining_types.pop()
                cluster_to_fraud[cluster_id] = fraud_type
                assigned_types.add(fraud_type)
                assigned_clusters.add(cluster_id)

    df['Pseudo_Labels'] = df[cluster_col].map(cluster_to_fraud)

    # display summary 
    for cluster_id, fraud_type in cluster_to_fraud.items():
        print(f"Cluster {cluster_id} assigned to : {fraud_type}")

    return df, cluster_to_fraud