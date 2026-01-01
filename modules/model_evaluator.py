# ============================================================================
# MODÜL 9: MODEL DEĞERLENDİRME
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, X_val, Y_val):
    """
    Modeli değerlendirir.
    
    Parametreler:
    ------------
    model : keras.Model
        Değerlendirilecek model
    X_val : numpy array
        Doğrulama görüntüleri
    Y_val : numpy array
        Doğrulama etiketleri
    
    Döndürür:
    --------
    dict
        Değerlendirme metrikleri
    """
    print("\n" + "="*60)
    print("MODÜL 9: MODEL DEĞERLENDİRME")
    print("="*60)
    
    # Model değerlendirme
    print("\nModel değerlendiriliyor...")
    loss, accuracy = model.evaluate(X_val, Y_val, verbose=0)
    
    print(f"\n✓ Model Değerlendirme Sonuçları:")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Tahmin yapma
    print("\nTahminler yapılıyor...")
    Y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true_classes = np.argmax(Y_val, axis=1)
    
    # Confusion Matrix
    print("\nConfusion Matrix oluşturuluyor...")
    cm = confusion_matrix(Y_true_classes, Y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Gerçek Etiket', fontsize=12)
    plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(Y_true_classes, Y_pred_classes, 
                              target_names=[str(i) for i in range(10)]))
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': Y_pred,
        'predicted_classes': Y_pred_classes,
        'true_classes': Y_true_classes
    }

