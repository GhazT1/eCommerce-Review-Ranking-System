import logging
import pandas as pd
from joblib import load
from collections import Counter
from config.config import Config

logger = logging.getLogger(__name__)

class Ranker:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.MODEL_PATH
        self.classifier = self._load_model()

    def _load_model(self):
        try:
            model = load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def rank_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank reviews within each product group using batch pairwise comparisons."""
        df['win'] = 0
        df['lose'] = 0
        df['review_score'] = 0.0
        
        feature_cols = ['review_len', 'Rn', 'Rp', 'Rs', 'Rc', 'Rd', 'Rsc']
        
        for product in df['product'].unique():
            product_df = df[df['product'] == product].copy()
            n_reviews = len(product_df)
            
            if n_reviews <= 1:
                df.loc[product_df.index, 'review_score'] = 1.0
                continue
                
            logger.info(f"Ranking {n_reviews} reviews for product: {product}")
            
            # Prepare batch pairwise comparisons
            # For each review, compare it against all other reviews in the same product group
            for idx in product_df.index:
                review_features = product_df.loc[idx, feature_cols].to_frame().T
                other_reviews = product_df[product_df.index != idx][feature_cols]
                
                # Create pairs: [Review A, Review B] for all B in others
                # In the original code, this was done via pd.merge on a dummy key 'j'
                # We'll replicate the logic but more efficiently
                
                # Repeat Review A features for each other review
                review_a_repeated = pd.concat([review_features] * len(other_reviews), ignore_index=True)
                other_reviews_reset = other_reviews.reset_index(drop=True)
                
                # Combine into a single feature vector for the classifier
                # Original code: pd.merge(C, D, how='outer', on='j')
                # This results in [A_features, B_features]
                comparison_batch = pd.concat([review_a_repeated, other_reviews_reset], axis=1)
                
                # Predict wins/losses in batch
                predictions = self.classifier.predict(comparison_batch.values)
                counts = Counter(predictions)
                
                wins = counts.get(1, 0)
                losses = counts.get(0, 0)
                
                df.at[idx, 'win'] = wins
                df.at[idx, 'lose'] = losses
                df.at[idx, 'review_score'] = float(wins) / n_reviews
                
        return df.sort_values(by=['product', 'review_score'], ascending=[True, False])
