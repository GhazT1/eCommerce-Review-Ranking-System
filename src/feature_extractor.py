import os
import pickle
import math
import logging
import spacy
import pandas as pd
import jellyfish
from langdetect import detect
from textblob import TextBlob, Word
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.metrics import edit_distance
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from config.config import Config

# Initialize logging
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
        self.stemmer = PorterStemmer()
        self._load_dictionaries()
        self._load_gib_model()

    def _load_dictionaries(self):
        """Load all dictionary files into memory."""
        try:
            with open(Config.HINDI_SWEAR_PATH, 'r') as fp:
                data = fp.read().lower().split('\n')
                self.hindi_swear_words = set([x.split('~')[0].strip().lower() for x in data if x.strip()])
            
            with open(Config.ENGLISH_SWEAR_PATH, 'r') as fp:
                data = fp.read().lower().split('\n')
                self.english_swear_words = set([x.strip() for x in data if x.strip()])
            
            with open(Config.SERVICE_TAGGER_PATH, 'r') as fp:
                data = fp.read().lower().split('\n')
                self.service_tags = set([x.strip() for x in data if x.strip()])
            
            with open(Config.COMPANY_TAGS_PATH, 'r') as fp:
                data = fp.read().lower().split('\n')
                self.company_tags = set([x.strip() for x in data if x.strip()])
                
            logger.info("All dictionaries loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading dictionaries: {e}")
            raise

    def _load_gib_model(self):
        """Load the gibberish detection model."""
        try:
            with open(Config.GIB_MODEL_PATH, 'rb') as f:
                self.gib_model_data = pickle.load(f)
            logger.info("Gibberish model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading gibberish model: {e}")
            raise

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def is_gibberish(self, text: str) -> bool:
        accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
        pos = {char: idx for idx, char in enumerate(accepted_chars)}
        
        def normalize(line):
            return [c.lower() for c in line if c.lower() in accepted_chars]

        def ngram(n, l):
            filtered = normalize(l)
            for start in range(0, len(filtered) - n + 1):
                yield ''.join(filtered[start:start + n])

        def avg_transition_prob(l, log_prob_mat):
            log_prob = 0.0
            transition_ct = 0
            for a, b in ngram(2, l):
                log_prob += log_prob_mat[pos[a]][pos[b]]
                transition_ct += 1
            return math.exp(log_prob / (transition_ct or 1))

        model_mat = self.gib_model_data['mat']
        threshold = self.gib_model_data['thresh']
        return avg_transition_prob(text, model_mat) < threshold

    def has_profanity(self, text: str) -> bool:
        words = set(text.lower().split())
        return not words.isdisjoint(self.hindi_swear_words) or not words.isdisjoint(self.english_swear_words)

    def has_competitive_brand(self, text: str) -> bool:
        words = text.lower().split()
        for word in words:
            for brand in self.company_tags:
                if jellyfish.damerau_levenshtein_distance(word, brand) <= Config.WORD_DISTANCE:
                    return True
        return False

    def get_sentiment_scores(self, text: str) -> dict:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'compound': self.analyzer.polarity_scores(text)['compound']
        }

    def get_service_tag_score(self, text: str) -> int:
        words = text.lower().split()
        for word in words:
            for tag in self.service_tags:
                if edit_distance(word, tag) <= 1:
                    return 1
        return 0

    def get_noun_scores(self, corpus: list) -> pd.Series:
        noun_tags = []
        processed_corpus = []
        
        for review in corpus:
            doc = self.nlp(review)
            # Extract nouns
            nouns = [self.stemmer.stem(token.lemma_) for token in doc 
                     if token.pos_ == "NOUN" and not token.is_stop and not token.is_punct and token.is_alpha]
            noun_tags.append(nouns)
            
            # Preprocess corpus for TF-IDF
            processed_review = " ".join([self.stemmer.stem(token.lemma_) for token in doc 
                                        if not token.is_stop and not token.is_punct and token.is_alpha])
            processed_corpus.append(processed_review)

        tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
        features = tfidf.fit_transform(processed_corpus)
        df_tfidf = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names_out())
        
        row_sums = df_tfidf.sum(axis=1)
        tfidf_scores = []
        
        for i, nouns in enumerate(noun_tags):
            if row_sums[i] == 0:
                tfidf_scores.append(0.0)
                continue
            
            noun_sum = sum(df_tfidf.at[i, n] for n in nouns if n in df_tfidf.columns)
            tfidf_scores.append(float(noun_sum / row_sums[i]))
            
        return pd.Series(tfidf_scores)
