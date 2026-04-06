import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    SRC_DIR = BASE_DIR / "src"
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    UTILS_DIR = SRC_DIR / "utils"
    DICT_DIR = UTILS_DIR / "DictionaryUtils"

    # File paths
    MODEL_PATH = MODELS_DIR / "randomforest.joblib"
    INPUT_FILE = DATA_DIR / "test.csv"
    OUTPUT_FILE = DATA_DIR / "test_ranked_output.csv"
    GIB_MODEL_PATH = UTILS_DIR / "gib_model.pki"

    # Dictionary paths
    HINDI_SWEAR_PATH = DICT_DIR / "hindi_swear_words.txt"
    ENGLISH_SWEAR_PATH = DICT_DIR / "english_profanity_google.txt"
    SERVICE_TAGGER_PATH = DICT_DIR / "service_tagger.txt"
    COMPANY_TAGS_PATH = DICT_DIR / "company_tags.txt"

    # Thresholds and parameters
    SPELL_THRESHOLD = 0.9
    WORD_DISTANCE = 1
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
