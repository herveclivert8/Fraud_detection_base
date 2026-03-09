import numpy as np


def preprocessing(df):

    df = df.copy()

    # temporal feature extraction 
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0 = lundi, 6 = dimanche

    # Log transformation
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Log_Amount_Diff'] = np.log1p(np.abs(df['Amount Paid'] - df['Amount Received']))

    # Dropping unnecessary columns
    drop_cols = ['From Account', 'To Account', 'Amount Paid', 'Amount Received', 'Timestamp']
    df.drop(columns=drop_cols, inplace=True)

    # From/To Bank : encodage by frequency
    for col in ['From Bank', 'To Bank']:
        freq = df[col].value_counts()
        df[col] = df[col].map(freq).fillna(0)
        df[col] = np.log1p(df[col])

    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    cat_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']

    return df, num_cols, cat_cols


def preprocessing_lgbm(df):
    df = df.copy()
    df, num_cols, cat_cols = preprocessing(df)

    for col in cat_cols:
        df[col] = df[col].astype('category')

    X = df.drop(columns='Pseudo_Labels')
    y = df['Pseudo_Labels']
    return X, y, cat_cols