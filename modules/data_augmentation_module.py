# ============================================================================
# MODÜL 4: VERİ ARTIRMA (DATA AUGMENTATION)
# ============================================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generator(rotation_range=5, width_shift_range=0.1, 
                         height_shift_range=0.1, zoom_range=0.1):
    """
    Veri artırma için ImageDataGenerator oluşturur.
    
    Parametreler:
    ------------
    rotation_range : int
        Döndürme açısı (derece)
    width_shift_range : float
        Yatay kaydırma oranı
    height_shift_range : float
        Dikey kaydırma oranı
    zoom_range : float
        Yakınlaştırma/uzaklaştırma oranı
    
    Döndürür:
    --------
    ImageDataGenerator
        Veri artırma generator'ı
    """
    print("\n" + "="*60)
    print("MODÜL 4: VERİ ARTIRMA GENERATOR'ı OLUŞTURMA")
    print("="*60)
    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=False,  # Rakam tanıma için uygun değil
        vertical_flip=False      # Rakam tanıma için uygun değil
    )
    
    print("✓ Veri artırma generator'ı oluşturuldu!")
    print(f"  - Döndürme: ±{rotation_range} derece")
    print(f"  - Yatay kaydırma: ±{width_shift_range*100:.1f}%")
    print(f"  - Dikey kaydırma: ±{height_shift_range*100:.1f}%")
    print(f"  - Yakınlaştırma: {1-zoom_range:.1%} - {1+zoom_range:.1%}")
    
    return datagen


def create_train_generator(datagen, X_train, Y_train, batch_size=32):
    """
    Eğitim için veri artırma generator'ı oluşturur.
    
    Parametreler:
    ------------
    datagen : ImageDataGenerator
        Veri artırma generator'ı
    X_train : numpy array
        Eğitim görüntüleri
    Y_train : numpy array
        Eğitim etiketleri
    batch_size : int
        Batch boyutu
    
    Döndürür:
    --------
    NumpyArrayIterator
        Eğitim generator'ı
    """
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
    print(f"✓ Eğitim generator'ı oluşturuldu (batch_size={batch_size})")
    return train_generator

