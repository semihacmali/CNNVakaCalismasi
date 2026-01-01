# ============================================================================
# MODÜL 6: CALLBACK'LER (ERKEN DURDURMA, ÖĞRENME ORANI AYARLAMA, vb.)
# ============================================================================
# 
# CALLBACK'LER NEDİR?
# -------------------
# Callback'ler, model eğitimi sırasında belirli noktalarda otomatik olarak
# çalışan fonksiyonlardır. Model eğitimini optimize etmek, izlemek ve kontrol
# etmek için kullanılırlar.
#
# BU MODÜLDE 3 CALLBACK KULLANILIYOR:
# ------------------------------------
# 1. EarlyStopping: Validasyon loss iyileşmezse eğitimi durdurur
#    - Overfitting'i önler
#    - Zaman tasarrufu sağlar
#    - En iyi modeli otomatik seçer
#
# 2. ReduceLROnPlateau: Validasyon loss iyileşmezse öğrenme oranını azaltır
#    - Model yakınsamaya yaklaştığında daha hassas optimizasyon yapar
#    - Küçük öğrenme oranı ile daha iyi sonuçlar alınabilir
#    - Otomatik öğrenme oranı ayarlama
#
# 3. ModelCheckpoint: En iyi performans gösteren modeli otomatik kaydeder
#    - Eğitim sırasında en iyi modeli korur
#    - Eğitim kesilirse en iyi model zaten kaydedilmiş olur
#    - Manuel müdahale gerektirmez
#
# DETAYLI AÇIKLAMA İÇİN: modules/CALLBACKS_ACIKLAMA.md dosyasına bakın
# ============================================================================

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def create_callbacks(patience=10, reduce_lr_patience=5, model_save_path='best_model.h5'):
    """
    Model eğitimi için callback'ler oluşturur.
    
    Bu fonksiyon, model eğitimini optimize etmek için 3 önemli callback oluşturur:
    1. EarlyStopping: Overfitting'i önler ve gereksiz eğitimi durdurur
    2. ReduceLROnPlateau: Öğrenme oranını otomatik ayarlar
    3. ModelCheckpoint: En iyi modeli otomatik kaydeder
    
    Parametreler:
    ------------
    patience : int
        Early stopping için sabır değeri
        - Kaç epoch boyunca iyileşme olmazsa eğitimi durduracak
        - Örnek: patience=10 → 10 epoch boyunca iyileşme yoksa durdur
        - Küçük veri setleri için: 5-10
        - Büyük veri setleri için: 10-20
    
    reduce_lr_patience : int
        Öğrenme oranı azaltma için sabır değeri
        - Kaç epoch boyunca iyileşme olmazsa öğrenme oranını azaltacak
        - Örnek: reduce_lr_patience=5 → 5 epoch boyunca iyileşme yoksa LR'yi yarıya indir
        - Genellikle Early Stopping'den daha küçük olmalı (örn: 5 vs 10)
        - Böylece önce LR azalır, sonra eğitim durur
    
    model_save_path : str
        En iyi modelin kaydedileceği dosya yolu
        - Örnek: 'best_model.h5', 'models/my_model.h5'
        - Model otomatik olarak bu yola kaydedilir
    
    Döndürür:
    --------
    list
        Callback listesi (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
        Bu liste model.fit() fonksiyonuna callbacks parametresi olarak verilir
    
    Örnek Kullanım:
    --------------
    >>> callbacks = create_callbacks(patience=10, reduce_lr_patience=5)
    >>> model.fit(X_train, Y_train, callbacks=callbacks, ...)
    
    Detaylı Açıklama:
    ----------------
    - EarlyStopping: monitor='val_loss' → Validasyon loss'u izler
                     patience=10 → 10 epoch bekler
                     restore_best_weights=True → En iyi ağırlıkları geri yükler
    
    - ReduceLROnPlateau: monitor='val_loss' → Validasyon loss'u izler
                        factor=0.5 → Öğrenme oranını yarıya indirir
                        patience=5 → 5 epoch bekler
                        min_lr=0.00001 → Minimum öğrenme oranı
    
    - ModelCheckpoint: monitor='val_accuracy' → Validasyon accuracy'yi izler
                      save_best_only=True → Sadece en iyi modeli kaydeder
    """
    print("\n" + "="*60)
    print("MODÜL 6: CALLBACK'LER OLUŞTURMA")
    print("="*60)
    
    # 1. EARLY STOPPING CALLBACK
    # ---------------------------
    # Ne yapar: Validasyon loss iyileşmezse eğitimi durdurur
    # Neden önemli: Overfitting'i önler, zaman tasarrufu sağlar
    # Nasıl çalışır: 
    #   - Her epoch'ta val_loss'u kontrol eder
    #   - Eğer val_loss patience kadar epoch boyunca iyileşmezse durdurur
    #   - restore_best_weights=True ile en iyi ağırlıkları geri yükler
    early_stopping = EarlyStopping(
        monitor='val_loss',              # İzlenecek metrik: validasyon loss
        patience=patience,              # Kaç epoch bekleyecek (varsayılan: 10)
        restore_best_weights=True,      # En iyi ağırlıkları geri yükle
        verbose=1                        # Bilgilendirme mesajlarını göster
    )
    
    # 2. REDUCE LEARNING RATE ON PLATEAU CALLBACK
    # --------------------------------------------
    # Ne yapar: Validasyon loss iyileşmezse öğrenme oranını azaltır
    # Neden önemli: Model yakınsamaya yaklaştığında daha hassas optimizasyon
    # Nasıl çalışır:
    #   - Her epoch'ta val_loss'u kontrol eder
    #   - Eğer val_loss patience kadar epoch boyunca iyileşmezse LR'yi azaltır
    #   - factor=0.5 → LR'yi yarıya indirir (örn: 0.001 → 0.0005)
    #   - min_lr=0.00001 → LR bu değerin altına düşmez
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',              # İzlenecek metrik: validasyon loss
        factor=0.5,                     # Öğrenme oranını ne kadar azaltacak (0.5 = yarıya indir)
        patience=reduce_lr_patience,    # Kaç epoch bekleyecek (varsayılan: 5)
        min_lr=0.00001,                 # Minimum öğrenme oranı (daha fazla azaltma yapılmaz)
        verbose=1                       # Bilgilendirme mesajlarını göster
    )
    
    # 3. MODEL CHECKPOINT CALLBACK
    # -----------------------------
    # Ne yapar: En iyi performans gösteren modeli otomatik kaydeder
    # Neden önemli: Eğitim sırasında en iyi modeli korur, güvenlik sağlar
    # Nasıl çalışır:
    #   - Her epoch'ta val_accuracy'yi kontrol eder
    #   - Eğer yeni bir en iyi accuracy görürse modeli kaydeder
    #   - save_best_only=True → Sadece en iyi modeli kaydeder (disk alanı tasarrufu)
    model_checkpoint = ModelCheckpoint(
        model_save_path,                # Modelin kaydedileceği dosya yolu
        monitor='val_accuracy',         # İzlenecek metrik: validasyon accuracy
        save_best_only=True,            # Sadece en iyi modeli kaydet
        verbose=1                       # Bilgilendirme mesajlarını göster
    )
    
    # Callback'leri listeye ekliyoruz
    callbacks = [
        early_stopping,     # 1. Erken durdurma
        reduce_lr,          # 2. Öğrenme oranı azaltma
        model_checkpoint    # 3. Model kaydetme
    ]
    
    print("✓ Callback'ler oluşturuldu:")
    print(f"  - Early Stopping (patience={patience})")
    print(f"    → {patience} epoch boyunca iyileşme olmazsa eğitimi durdurur")
    print(f"  - Learning Rate Reduction (patience={reduce_lr_patience})")
    print(f"    → {reduce_lr_patience} epoch boyunca iyileşme olmazsa LR'yi yarıya indirir")
    print(f"  - Model Checkpoint (kayıt yolu: {model_save_path})")
    print(f"    → En iyi validasyon accuracy gösteren modeli kaydeder")
    
    return callbacks

