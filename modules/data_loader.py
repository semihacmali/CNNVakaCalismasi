# ============================================================================
# MODÜL 2: VERİ YÜKLEME VE ÖN İŞLEME
# ============================================================================

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(train_path="data/train.csv", test_path="data/test.csv", 
                             validation_split=0.1, random_state=13):
    """
    Veri setini yükler ve ön işleme yapar.
    
    Parametreler:
    ------------
    train_path : str
        Eğitim verisi dosya yolu
    test_path : str
        Test verisi dosya yolu
    validation_split : float
        Doğrulama seti için ayrılacak oran (varsayılan: 0.1 = %10)
    random_state : int
        Rastgelelik için seed değeri
    
    Döndürür:
    --------
    X_train, X_val, Y_train, Y_val, X_test : numpy arrays
        Ön işlenmiş veri setleri
    """
    print("\n" + "="*60)
    print("MODÜL 2: VERİ YÜKLEME VE ÖN İŞLEME")
    print("="*60)
    
    # 1. Veri yükleme
    print("\n1. Veri yükleme işlemi başlatılıyor...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"   ✓ Eğitim verisi yüklendi: {train.shape}")
    print(f"   ✓ Test verisi yüklendi: {test.shape}")
    
    # 2. Etiket ve özellik ayırma
    print("\n2. Etiket ve özellik ayırma işlemi...")
    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)
    
    print(f"   ✓ Etiket sayısı: {len(Y_train)}")
    print(f"   ✓ Özellik sayısı: {X_train.shape[1]} (784 piksel = 28x28)")
    
    # 3. Normalizasyon (0-255 aralığından 0-1 aralığına)
    print("\n3. Normalizasyon işlemi (0-255 -> 0-1)...")
    X_train = X_train / 255.0
    test = test / 255.0
    
    print(f"   ✓ Eğitim verisi normalizasyonu tamamlandı")
    print(f"   ✓ Test verisi normalizasyonu tamamlandı")
    print(f"   ✓ Piksel değer aralığı: [{X_train.min().min():.3f}, {X_train.max().max():.3f}]")
    
    # 4. Yeniden şekillendirme (CNN için: örnek_sayısı, yükseklik, genişlik, kanal)
    print("\n4. Yeniden şekillendirme işlemi (CNN formatına dönüştürme)...")
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    X_test = test.values.reshape(-1, 28, 28, 1)
    
    print(f"   ✓ Eğitim verisi: {X_train.shape}")
    print(f"   ✓ Test verisi: {X_test.shape}")
    
    # 5. Etiket kodlama (One-Hot Encoding)
    print("\n5. Etiket kodlama işlemi (One-Hot Encoding)...")
    num_classes = 10
    Y_train = to_categorical(Y_train, num_classes=num_classes)
    
    print(f"   ✓ Etiketler one-hot encoding formatına dönüştürüldü")
    print(f"   ✓ Etiket boyutu: {Y_train.shape}")
    
    # 6. Eğitim ve doğrulama seti ayırma
    print(f"\n6. Eğitim ve doğrulama seti ayırma (%{validation_split*100} doğrulama)...")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, 
        test_size=validation_split, 
        random_state=random_state
    )
    
    print(f"   ✓ Eğitim seti: {X_train.shape}")
    print(f"   ✓ Doğrulama seti: {X_val.shape}")
    print(f"   ✓ Eğitim etiketleri: {Y_train.shape}")
    print(f"   ✓ Doğrulama etiketleri: {Y_val.shape}")
    
    print("\n" + "="*60)
    print("VERİ YÜKLEME VE ÖN İŞLEME TAMAMLANDI!")
    print("="*60)
    
    return X_train, X_val, Y_train, Y_val, X_test

