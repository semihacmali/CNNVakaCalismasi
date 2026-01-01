# ============================================================================
# MODÜL 7: MODEL EĞİTİMİ
# ============================================================================


def train_model(model, X_train, Y_train, X_val, Y_val,
                train_generator=None, 
                epochs=50, steps_per_epoch=None, batch_size=32,
                use_augmentation=True, callbacks=None):
    """
    Modeli eğitir.
    
    Parametreler:
    ------------
    model : keras.Model
        Eğitilecek model
    X_train : numpy array
        Eğitim görüntüleri
    Y_train : numpy array
        Eğitim etiketleri
    X_val : numpy array
        Doğrulama görüntüleri
    Y_val : numpy array
        Doğrulama etiketleri
    train_generator : NumpyArrayIterator, optional
        Eğitim veri generator'ı (augmentation ile)
    epochs : int
        Epoch sayısı
    steps_per_epoch : int
        Her epoch'ta işlenecek batch sayısı (None ise otomatik hesaplanır)
    batch_size : int
        Batch boyutu
    use_augmentation : bool
        Veri artırma kullanılsın mı?
    callbacks : list
        Callback listesi
    
    Döndürür:
    --------
    keras.callbacks.History
        Eğitim geçmişi
    """
    print("\n" + "="*60)
    print("MODÜL 7: MODEL EĞİTİMİ")
    print("="*60)
    
    if steps_per_epoch is None:
        # Eğitim verisi boyutundan steps_per_epoch hesapla
        if use_augmentation and train_generator is not None:
            steps_per_epoch = len(train_generator) if hasattr(train_generator, '__len__') else X_train.shape[0] // batch_size
        else:
            steps_per_epoch = X_train.shape[0] // batch_size
    
    print(f"\nEğitim parametreleri:")
    print(f"  - Epoch sayısı: {epochs}")
    print(f"  - Batch boyutu: {batch_size}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Veri artırma: {'Evet' if use_augmentation else 'Hayır'}")
    
    if use_augmentation and train_generator is not None:
        # Veri artırma ile eğitim
        print("\nVeri artırma ile eğitim başlatılıyor...")
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Normal eğitim (veri artırma olmadan)
        print("\nNormal eğitim başlatılıyor (veri artırma olmadan)...")
        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    print("\n✓ Model eğitimi tamamlandı!")
    
    return history

