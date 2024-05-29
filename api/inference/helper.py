from keras.models import load_model
import pandas as pd
import numpy as np
import pickle

def ndcg_at_k(y_true, y_pred, k=5):
    def compute_dcg(y_true, y_pred, k):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(1, len(y_true) + 1) + 1)
        return np.sum(gains / discounts)

    def compute_ndcg(y_true, y_pred, k):
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        if len(np.unique(y_true_np)) < 2:
            return 0.0
        dcg = compute_dcg(y_true_np, y_pred_np, k)
        idcg = compute_dcg(y_true_np, y_true_np, k)
        return dcg / idcg if idcg > 0 else 0.0

    return tf.py_function(compute_ndcg, (y_true, y_pred, k), tf.double)

def ndcg_5(y_true, y_pred):
    return ndcg_at_k(y_true, y_pred, k=5)

def ndcg_10(y_true, y_pred):
    return ndcg_at_k(y_true, y_pred, k=10)


def mean_mrr(y_true, y_pred):
    def compute_mrr(y_true, y_pred):
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        if len(np.unique(y_true_np)) < 2:
            return 0.0
        order = np.argsort(y_pred_np)[::-1]
        y_true_sorted = np.take(y_true_np, order)
        rr = [1.0 / (i + 1) for i, x in enumerate(y_true_sorted) if x == 1]
        return np.mean(rr) if rr else 0.0
    return tf.py_function(compute_mrr, (y_true, y_pred), tf.double)


def g_auc(y_true, y_pred):
    def compute_auc(y_true, y_pred):
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        if len(np.unique(y_true_np)) < 2:
            return 0.0  # Explicitly return 0.0 or another appropriate value
        try:
            return roc_auc_score(y_true_np, y_pred_np)
        except ValueError:
            return 0.0
    return tf.py_function(compute_auc, (y_true, y_pred), tf.double)

model = load_model('./model_data/content-based-reduced.h5', custom_objects={'ndcg_5': ndcg_5, 'ndcg_10': ndcg_10, 'mean_mrr': mean_mrr, 'g_auc': g_auc})

with open('./model_data/embeddings_dict.pkl', 'rb') as f:
   embeddings_dict = pickle.load(f)
   
# Load DataFrames from disk
user_profiles_df_all = pd.read_pickle("./model_data/user_profiles_df_all-reduced.pkl")
df_articles = pd.read_pickle("./model_data/df_articles-reduced.pkl")
article_embeddings_df = pd.read_pickle("./model_data/article_embeddings_df-reduced.pkl")

def infer_all_articles_scores(user_id, df, df_articles, article_embeddings_df, model):
    
    # Retrieve the user's embedding
    user_profile = df[df['user_id'] == user_id].iloc[0]
    if user_profile.empty:
        raise ValueError("User ID not found in the user profiles.")

    user_embedding = user_profile['user_embedding']

    # Get all articles embeddings
    embeddings_dict = article_embeddings_df.T.to_dict('list')
    
    article_ids = list(embeddings_dict.keys())
    combined_features_list = [np.concatenate((user_embedding, article_embedding)).reshape(1, -1) 
                              for article_embedding in embeddings_dict.values()]

    all_embeddings = np.vstack(combined_features_list)
    
    # Predict relevance scores using the trained model
    scores = model.predict(all_embeddings, verbose=0).flatten()

    # Create a dataframe with article IDs, category IDs, and scores
    article_scores_df = df_articles[['article_id', 'category_id']].copy()
    article_scores_df['score'] = article_scores_df['article_id'].map(dict(zip(article_ids, scores)))
    
    # Remove any unwanted header rows if present
    # article_scores_df.columns = article_scores_df.columns.droplevel(0)
    article_scores_df.reset_index(drop=True, inplace=True)
    article_scores_df.sort_values(by='score', ascending=False, inplace=True)
    
    article_scores_df_remove_seen = article_scores_df.copy()
    article_scores_df_remove_seen = article_scores_df_remove_seen[~article_scores_df_remove_seen["article_id"].isin(user_profile["click_article_id"])]

    result = {
        "top_recommendation": article_scores_df_remove_seen[:5].to_dict(orient="records"),
        "clicked_articles": article_scores_df[article_scores_df["article_id"].isin(user_profile["click_article_id"])].to_dict(orient="records")
    }
    
    return result

def inference(user_id):
    return infer_all_articles_scores(user_id, user_profiles_df_all, df_articles, article_embeddings_df, model)
