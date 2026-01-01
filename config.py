# ============================================================================
# YAPILANDIRMA DOSYASI
# ============================================================================
# Tüm modüller için ortak ayarlar burada tanımlanır

# Veri yolları
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Model ayarları
MODEL_TYPE = 'semih'  # 'simple', 'standard', 'deep', 'semih'
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

# Eğitim ayarları
EPOCHS = 3
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 13

# Veri artırma ayarları
USE_AUGMENTATION = True
ROTATION_RANGE = 5
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.1

# Callback ayarları
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Dosya yolları
MODEL_SAVE_PATH = MODEL_TYPE + 'best_model.h5'
SUBMISSION_PATH = MODEL_TYPE + 'submission.csv'

