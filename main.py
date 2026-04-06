import sys
import os
import logging
import time
import argparse
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config
from feature_extractor import FeatureExtractor
from ranker import Ranker

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

class ReviewPipeline:
    def __init__(self, input_file: str = None, output_file: str = None):
        self.input_file = input_file or Config.INPUT_FILE
        self.output_file = output_file or Config.OUTPUT_FILE
        self.extractor = FeatureExtractor()
        self.ranker = Ranker()

    def run(self):
        start_time = time.time()
        logger.info(f"Starting pipeline for file: {self.input_file}")

        try:
            # 1. Load Data
            df = pd.read_csv(self.input_file)
            logger.info(f"Loaded {len(df)} reviews.")

            # 2. Preprocessing & Filtering
            df = self._preprocess(df)
            logger.info(f"Preprocessing complete. {len(df)} reviews remaining.")

            # 3. Feature Engineering
            df = self._extract_features(df)
            logger.info("Feature engineering complete.")

            # 4. Ranking
            df = self.ranker.rank_reviews(df)
            logger.info("Ranking complete.")

            # 5. Save Results
            df[['product', 'answer_option', 'review_score']].to_csv(self.output_file, index=False)
            logger.info(f"Results saved to {self.output_file}")

            total_time = time.time() - start_time
            logger.info(f"Pipeline finished successfully in {total_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out bad reviews based on language, gibberish, profanity, and brands."""
        bad_indices = set()
        stats = {"language": 0, "gibberish": 0, "profanity": 0, "brand": 0}

        for idx, row in df.iterrows():
            review = row['answer_option']
            
            # Language check
            lang = self.extractor.detect_language(review)
            if lang in ['hi', 'mr']:
                bad_indices.add(idx)
                stats["language"] += 1
                continue

            # Gibberish check
            if self.extractor.is_gibberish(review):
                bad_indices.add(idx)
                stats["gibberish"] += 1
                continue

            # Profanity check
            if self.extractor.has_profanity(review):
                bad_indices.add(idx)
                stats["profanity"] += 1
                continue

            # Competitive brand check
            if self.extractor.has_competitive_brand(review):
                bad_indices.add(idx)
                stats["brand"] += 1
                continue

        logger.info(f"Filtering stats: {stats}")
        return df.drop(index=list(bad_indices)).reset_index(drop=True)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all required features for the ranking model."""
        # Basic features
        df['review_len'] = df['answer_option'].apply(lambda x: len(x.split()))
        
        # Sentiment and service features
        sentiments = df['answer_option'].apply(self.extractor.get_sentiment_scores)
        df['Rp'] = sentiments.apply(lambda x: x['polarity'])
        df['Rs'] = sentiments.apply(lambda x: x['subjectivity'])
        df['Rsc'] = sentiments.apply(lambda x: x['compound'])
        df['Rd'] = df['answer_option'].apply(self.extractor.get_service_tag_score)

        # Noun scores (requires corpus-level processing)
        df['Rn'] = self.extractor.get_noun_scores(df['answer_option'].tolist())

        # Relative complexity (Rc)
        for product in df['product'].unique():
            product_mask = df['product'] == product
            product_reviews = df.loc[product_mask, 'answer_option']
            
            unique_words = set()
            for r in product_reviews:
                unique_words.update(r.lower().split())
            
            vocab_size = len(unique_words) or 1
            df.loc[product_mask, 'Rc'] = df.loc[product_mask, 'answer_option'].apply(
                lambda x: len(set(x.lower().split())) / vocab_size
            )

        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production-grade Review Ranking Pipeline")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--output", type=str, help="Path to output CSV file")
    args = parser.parse_args()

    pipeline = ReviewPipeline(input_file=args.input, output_file=args.output)
    pipeline.run()
