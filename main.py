# ============================================================================
# ANA PROGRAM - CNN MODELİ PIPELINE
# ============================================================================
# Bu dosya, tüm modülleri birleştirerek tam pipeline'ı çalıştırır
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

# Modülleri import ediyoruz
from modules.data_loader import load_and_preprocess_data
from modules.data_visualization import visualize_data
from modules.data_augmentation_module import create_data_generator, create_train_generator
from modules.model_builder import create_cnn_model
from modules.callbacks import create_callbacks
from modules.model_trainer import train_model
from modules.training_visualizer import plot_training_history
from modules.model_evaluator import evaluate_model
from modules.predictor import make_predictions

# Yapılandırma dosyasını import ediyoruz
import config

# GPU kullanılabilirliğini kontrol ediyoruz
print("="*80)
print("CNN MODELİ - TAM PIPELINE")
print("="*80)
print(f"TensorFlow versiyonu: {tf.__version__}")
print(f"GPU kullanılabilir mi? {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print(f"GPU cihazları: {tf.config.list_physical_devices('GPU')}")
print("="*80)


def run_complete_pipeline(train_path=None, 
                         test_path=None,
                         model_type=None,
                         epochs=None,
                         batch_size=None,
                         use_augmentation=None,
                         save_model_path=None,
                         submission_path=None):
    """
    Tüm pipeline'ı çalıştırır: veri yükleme, model oluşturma, eğitme, değerlendirme ve tahmin.
    
    Parametreler:
    ------------
    train_path : str, optional
        Eğitim verisi dosya yolu (None ise config'den alınır)
    test_path : str, optional
        Test verisi dosya yolu (None ise config'den alınır)
    model_type : str, optional
        Model tipi: 'simple', 'standard', 'deep' (None ise config'den alınır)
    epochs : int, optional
        Epoch sayısı (None ise config'den alınır)
    batch_size : int, optional
        Batch boyutu (None ise config'den alınır)
    use_augmentation : bool, optional
        Veri artırma kullanılsın mı? (None ise config'den alınır)
    save_model_path : str, optional
        Model kayıt yolu (None ise config'den alınır)
    submission_path : str, optional
        Submission dosyası kayıt yolu (None ise config'den alınır)
    """
    # Yapılandırma değerlerini kullan (parametre verilmezse)
    train_path = train_path or config.TRAIN_PATH
    test_path = test_path or config.TEST_PATH
    model_type = model_type or config.MODEL_TYPE
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    use_augmentation = use_augmentation if use_augmentation is not None else config.USE_AUGMENTATION
    save_model_path = save_model_path or config.MODEL_SAVE_PATH
    submission_path = submission_path or config.SUBMISSION_PATH
    
    print("\n" + "="*80)
    print("CNN MODELİ - TAM PIPELINE")
    print("="*80)
    print(f"\nYapılandırma:")
    print(f"  - Model tipi: {model_type}")
    print(f"  - Epoch sayısı: {epochs}")
    print(f"  - Batch boyutu: {batch_size}")
    print(f"  - Veri artırma: {'Evet' if use_augmentation else 'Hayır'}")
    print(f"  - Model kayıt yolu: {save_model_path}")
    print(f"  - Submission yolu: {submission_path}")
    print("="*80)
    
    # 1. Veri yükleme ve ön işleme
    X_train, X_val, Y_train, Y_val, X_test = load_and_preprocess_data(
        train_path, test_path,
        validation_split=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_STATE
    )
    
    # 2. Veri görselleştirme
    visualize_data(X_train, Y_train)
    
    # 3. Veri artırma (isteğe bağlı)
    if use_augmentation:
        datagen = create_data_generator(
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT_RANGE,
            height_shift_range=config.HEIGHT_SHIFT_RANGE,
            zoom_range=config.ZOOM_RANGE
        )
        train_generator = create_train_generator(datagen, X_train, Y_train, batch_size)
    else:
        train_generator = None
    
    # 4. Model oluşturma
    model = create_cnn_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        model_type=model_type
    )
    
    # 5. Callback'ler oluşturma
    callbacks = create_callbacks(
        patience=config.EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=config.REDUCE_LR_PATIENCE,
        model_save_path=save_model_path
    )
    
    # 6. Model eğitimi
    history = train_model(
        model, X_train, Y_train, X_val, Y_val,
        train_generator=train_generator,
        epochs=epochs,
        batch_size=batch_size,
        use_augmentation=use_augmentation,
        callbacks=callbacks
    )
    
    # 7. Eğitim geçmişini görselleştirme
    plot_training_history(history)
    
    # 8. Model değerlendirme
    evaluation_results = evaluate_model(model, X_val, Y_val)
    
    # 9. Tahmin ve submission
    predictions = make_predictions(model, X_test, submission_path)
    
    print("\n" + "="*80)
    print("TÜM İŞLEMLER TAMAMLANDI!")
    print("="*80)
    print(f"\nÖzet:")
    print(f"  ✓ Model eğitildi ve kaydedildi: {save_model_path}")
    print(f"  ✓ Doğruluk: {evaluation_results['accuracy']*100:.2f}%")
    print(f"  ✓ Submission dosyası oluşturuldu: {submission_path}")
    print("="*80)
    
    return model, history, evaluation_results, predictions


if __name__ == "__main__":
    # Ana program çalıştırılıyor
    # Tüm parametreler config.py dosyasından alınır
    # İsterseniz burada parametreleri özelleştirebilirsiniz
    
    model, history, results, predictions = run_complete_pipeline(
        # Özelleştirmek için parametreleri buraya ekleyebilirsiniz:
        # model_type='standard',  # 'simple', 'standard', 'deep'
        # epochs=50,
        # batch_size=32,
        # use_augmentation=True,
        # save_model_path='best_model.h5',
        # submission_path='submission.csv'
    )

