# ============================================================================
# MODÜL 3: VERİ GÖRSELLEŞTİRME
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_data(X_train, Y_train, num_samples=10):
    """
    Veri setinden örnek görüntüleri görselleştirir.
    
    Parametreler:
    ------------
    X_train : numpy array
        Eğitim görüntüleri
    Y_train : numpy array
        Eğitim etiketleri (one-hot encoded)
    num_samples : int
        Gösterilecek örnek sayısı
    """
    print("\n" + "="*60)
    print("MODÜL 3: VERİ GÖRSELLEŞTİRME")
    print("="*60)
    
    # Etiket dağılımını görselleştirme
    plt.figure(figsize=(15, 7))
    labels = np.argmax(Y_train, axis=1)
    g = sns.countplot(x=labels, palette="icefire")
    plt.title("Eğitim Verisi Sınıf Dağılımı", fontsize=16)
    plt.xlabel("Rakam Sınıfı", fontsize=12)
    plt.ylabel("Örnek Sayısı", fontsize=12)
    plt.show()
    
    # Örnek görüntüleri görselleştirme
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Örnek Eğitim Görüntüleri", fontsize=16, fontweight='bold')
    
    for i in range(min(num_samples, 10)):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(X_train[i][:, :, 0], cmap='gray')
        true_label = np.argmax(Y_train[i])
        axes[row, col].set_title(f"Etiket: {true_label}", fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Veri görselleştirme tamamlandı!")

