# ============================================================================
# MODÜL 8: EĞİTİM GEÇMİŞİNİ GÖRSELLEŞTİRME
# ============================================================================

import matplotlib.pyplot as plt


def plot_training_history(history):
    """
    Eğitim geçmişini görselleştirir.
    
    Parametreler:
    ------------
    history : keras.callbacks.History
        Eğitim geçmişi
    """
    print("\n" + "="*60)
    print("MODÜL 8: EĞİTİM GEÇMİŞİNİ GÖRSELLEŞTİRME")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy grafiği
    axes[0].plot(history.history['accuracy'], label='Eğitim Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss grafiği
    axes[1].plot(history.history['loss'], label='Eğitim Loss')
    axes[1].plot(history.history['val_loss'], label='Doğrulama Loss')
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Eğitim geçmişi görselleştirildi!")

