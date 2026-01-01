# ============================================================================
# MODÜL 10: TAHMİN VE SUBMISSION
# ============================================================================

import numpy as np
import pandas as pd


def make_predictions(model, X_test, submission_path='submission.csv'):
    """
    Test verisi üzerinde tahmin yapar ve submission dosyası oluşturur.
    
    Parametreler:
    ------------
    model : keras.Model
        Tahmin yapacak model
    X_test : numpy array
        Test görüntüleri
    submission_path : str
        Submission dosyası kayıt yolu
    
    Döndürür:
    --------
    numpy array
        Tahmin edilen sınıflar
    """
    print("\n" + "="*60)
    print("MODÜL 10: TAHMİN VE SUBMISSION")
    print("="*60)
    
    print(f"\nTest verisi üzerinde tahmin yapılıyor...")
    print(f"Test verisi boyutu: {X_test.shape}")
    
    # Tahmin yapma
    predictions = model.predict(X_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"\n✓ Tahminler tamamlandı!")
    print(f"  - Toplam tahmin sayısı: {len(predicted_classes)}")
    print(f"  - Tahmin dağılımı:")
    unique, counts = np.unique(predicted_classes, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    Rakam {label}: {count} örnek ({count/len(predicted_classes)*100:.2f}%)")
    
    # Submission dosyası oluşturma
    print(f"\nSubmission dosyası oluşturuluyor: {submission_path}")
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_classes) + 1),
        'Label': predicted_classes
    })
    submission.to_csv(submission_path, index=False)
    
    print(f"✓ Submission dosyası kaydedildi: {submission_path}")
    
    return predicted_classes

