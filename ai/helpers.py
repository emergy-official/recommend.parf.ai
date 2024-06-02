import os
import pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# Read all csv from clicks folder    
def read_csv(file_path):
    return pd.read_csv(file_path)

def load_dataset():
    # Load datasets
    df_articles = pd.read_csv('input/archive/articles_metadata.csv')
    df_clicks_sample = pd.read_csv('input/archive/clicks_sample.csv')
    folder_path = 'input/archive/clicks'

    with open('input/archive/articles_embeddings.pickle', 'rb') as file:
        article_embeddings = pickle.load(file)
        
    csv_files_clicks = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    with ProcessPoolExecutor() as executor:
        df_clicks = list(executor.map(read_csv, csv_files_clicks))

    df_clicks = pd.concat(df_clicks)
    df_clicks.reset_index(drop=True, inplace=True)
    
    return df_articles, df_clicks, article_embeddings.astype(np.float32)

def preprocessing_articles(df_articles):
    df_articles['created_at_dt'] = pd.to_datetime(df_articles['created_at_ts'], unit='ms')
    df_articles.drop(columns=['created_at_ts'], inplace=True)
    
    return df_articles

def preprocessing_clicks(df_clicks):
    df_clicks['session_start_dt'] = pd.to_datetime(df_clicks['session_start'], unit='ms')
    df_clicks['click_timestamp_dt'] = pd.to_datetime(df_clicks['click_timestamp'], unit='ms')

    # Drop original timestamp columns if no longer needed
    df_clicks.drop(columns=['session_start', 'click_timestamp'], inplace=True)

    # 3. Extract additional time features
    df_clicks['click_hour'] = df_clicks['click_timestamp_dt'].dt.hour
    df_clicks['click_dayofweek'] = df_clicks['click_timestamp_dt'].dt.dayofweek
    
    return df_clicks

# Function to process a single user profile
def process_user_profile(user, embeddings_dict, articles_df):
    X_user = []
    y_user = []
    
    user_embedding = user['user_embedding']
    clicked_articles = user['click_article_id']
    
    for article_id in clicked_articles:
        if article_id in embeddings_dict:
            article_embedding = embeddings_dict[article_id]
            combined_features = np.concatenate((user_embedding, article_embedding))
            X_user.append(combined_features)
            y_user.append(1)  # Positive sample
    
    # Add some negative samples for training
    negative_samples = articles_df[~articles_df['article_id'].isin(clicked_articles)]['article_id'].sample(n=len(clicked_articles))
    
    for article_id in negative_samples:
        if article_id in embeddings_dict:
            article_embedding = embeddings_dict[article_id]
            combined_features = np.concatenate((user_embedding, article_embedding))
            X_user.append(combined_features)
            y_user.append(0)  # Negative sample
    
    return X_user, y_user

# Main function to prepare data using multi-CPU processing (the difference is not that visible)
def prepare_data(user_profiles_df_train, articles_df, articles_embeddings_df, max_users=500):
    embeddings_dict = articles_embeddings_df.T.to_dict('list')
    
    X = []
    y = []
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, user in tqdm(user_profiles_df_train.iterrows(), total=min(len(user_profiles_df_train), max_users)):
            # if i >= max_users:
            #     break
            futures.append(executor.submit(process_user_profile, user, embeddings_dict, articles_df))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            X_user, y_user = future.result()
            X.extend(X_user)
            y.extend(y_user)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y