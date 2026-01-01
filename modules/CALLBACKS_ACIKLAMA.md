# CALLBACK'LER NEDİR VE NE İŞE YARAR?

## 📚 Genel Bakış

**Callback'ler**, model eğitimi sırasında belirli noktalarda otomatik olarak çalışan fonksiyonlardır. Model eğitimini optimize etmek, izlemek ve kontrol etmek için kullanılırlar.

## 🎯 Callback'lerin Amacı

1. **Eğitimi Optimize Etmek**: Gereksiz epoch'ları önlemek, öğrenme oranını ayarlamak
2. **En İyi Modeli Kaydetmek**: Eğitim sırasında en iyi performans gösteren modeli otomatik kaydetmek
3. **Overfitting'i Önlemek**: Modelin eğitim verisine aşırı uyum sağlamasını engellemek
4. **Zaman ve Kaynak Tasarrufu**: Gereksiz eğitim süresini önlemek

---

## 🔍 1. EARLY STOPPING (Erken Durdurma)

### Ne İşe Yarar?

Early Stopping, model eğitimi sırasında validasyon loss'u (kayıp) iyileşmediğinde eğitimi otomatik olarak durdurur.

### Nasıl Çalışır?

```python
EarlyStopping(
    monitor='val_loss',           # İzlenecek metrik: validasyon loss
    patience=10,                  # 10 epoch boyunca iyileşme olmazsa durdur
    restore_best_weights=True,    # En iyi ağırlıkları geri yükle
    verbose=1                     # Bilgilendirme mesajlarını göster
)
```

### Örnek Senaryo:

```
Epoch 1: val_loss = 0.5
Epoch 2: val_loss = 0.4  ✓ İyileşti
Epoch 3: val_loss = 0.35  ✓ İyileşti
Epoch 4: val_loss = 0.36  ✗ Kötüleşti (patience başladı)
Epoch 5: val_loss = 0.37  ✗ Kötüleşti (patience: 1/10)
Epoch 6: val_loss = 0.38  ✗ Kötüleşti (patience: 2/10)
...
Epoch 15: val_loss = 0.45  ✗ Kötüleşti (patience: 10/10)
→ Eğitim durduruldu! En iyi model (Epoch 3) geri yüklendi.
```

### Neden Önemli?

- ✅ **Zaman Tasarrufu**: Gereksiz epoch'ları önler
- ✅ **Overfitting Önleme**: Model aşırı öğrenmeye başladığında durdurur
- ✅ **En İyi Model**: Otomatik olarak en iyi performans gösteren modeli seçer
- ✅ **Kaynak Tasarrufu**: CPU/GPU kullanımını optimize eder

### Parametreler:

- **monitor**: İzlenecek metrik (`'val_loss'`, `'val_accuracy'`, `'loss'`, vb.)
- **patience**: Kaç epoch bekleyecek (varsayılan: 10)
- **restore_best_weights**: En iyi ağırlıkları geri yükle (True/False)
- **verbose**: Bilgilendirme mesajları (0=sessiz, 1=mesajlar)

---

## 📉 2. REDUCE LR ON PLATEAU (Öğrenme Oranı Azaltma)

### Ne İşe Yarar?

Validasyon loss'u belirli bir süre iyileşmediğinde, öğrenme oranını (learning rate) otomatik olarak azaltır.

### Nasıl Çalışır?

```python
ReduceLROnPlateau(
    monitor='val_loss',        # İzlenecek metrik
    factor=0.5,                # Öğrenme oranını yarıya indir
    patience=5,                # 5 epoch bekleyip iyileşme yoksa azalt
    min_lr=0.00001,           # Minimum öğrenme oranı (daha fazla azaltma)
    verbose=1                  # Bilgilendirme mesajları
)
```

### Örnek Senaryo:

```
Başlangıç Learning Rate: 0.001

Epoch 1: val_loss = 0.5, LR = 0.001
Epoch 2: val_loss = 0.4, LR = 0.001  ✓ İyileşti
Epoch 3: val_loss = 0.35, LR = 0.001 ✓ İyileşti
Epoch 4: val_loss = 0.36, LR = 0.001 ✗ Kötüleşti (patience başladı)
Epoch 5: val_loss = 0.37, LR = 0.001 ✗ Kötüleşti (patience: 1/5)
Epoch 6: val_loss = 0.38, LR = 0.001 ✗ Kötüleşti (patience: 2/5)
Epoch 7: val_loss = 0.39, LR = 0.001 ✗ Kötüleşti (patience: 3/5)
Epoch 8: val_loss = 0.40, LR = 0.001 ✗ Kötüleşti (patience: 4/5)
Epoch 9: val_loss = 0.41, LR = 0.001 ✗ Kötüleşti (patience: 5/5)
→ Learning Rate azaltıldı: 0.001 → 0.0005

Epoch 10: val_loss = 0.35, LR = 0.0005 ✓ İyileşti (yeni LR ile)
```

### Neden Önemli?

- ✅ **İnce Ayar**: Model yakınsamaya yaklaştığında daha küçük adımlarla ilerler
- ✅ **Daha İyi Sonuçlar**: Küçük öğrenme oranı ile daha hassas optimizasyon
- ✅ **Otomatik Optimizasyon**: Manuel müdahale gerektirmez
- ✅ **Yerel Minimum'dan Çıkış**: Bazen daha küçük LR ile daha iyi sonuçlar alınır

### Parametreler:

- **monitor**: İzlenecek metrik (`'val_loss'`, `'val_accuracy'`, vb.)
- **factor**: Öğrenme oranını ne kadar azaltacak (0.5 = yarıya indir)
- **patience**: Kaç epoch bekleyecek (varsayılan: 5)
- **min_lr**: Minimum öğrenme oranı (daha fazla azaltma yapılmaz)
- **verbose**: Bilgilendirme mesajları

### Öğrenme Oranı Nedir?

Öğrenme oranı (Learning Rate), modelin her adımda ne kadar büyük değişiklik yapacağını belirler:
- **Yüksek LR (örn: 0.01)**: Büyük adımlar, hızlı öğrenme ama kararsız
- **Düşük LR (örn: 0.0001)**: Küçük adımlar, yavaş ama stabil öğrenme
- **Adaptif LR**: İhtiyaca göre otomatik ayarlanır (ReduceLROnPlateau)

---

## 💾 3. MODEL CHECKPOINT (Model Kaydetme)

### Ne İşe Yarar?

Eğitim sırasında belirli koşullar sağlandığında (örn: en iyi validasyon accuracy) modeli otomatik olarak kaydeder.

### Nasıl Çalışır?

```python
ModelCheckpoint(
    'best_model.h5',          # Kayıt dosya yolu
    monitor='val_accuracy',    # İzlenecek metrik: validasyon accuracy
    save_best_only=True,      # Sadece en iyi modeli kaydet
    verbose=1                 # Bilgilendirme mesajları
)
```

### Örnek Senaryo:

```
Epoch 1: val_accuracy = 0.85 → Model kaydedildi! (en iyi şimdilik)
Epoch 2: val_accuracy = 0.87 → Model kaydedildi! (daha iyi)
Epoch 3: val_accuracy = 0.89 → Model kaydedildi! (daha iyi)
Epoch 4: val_accuracy = 0.88 → Model kaydedilmedi (daha kötü)
Epoch 5: val_accuracy = 0.90 → Model kaydedildi! (en iyi)
...
Epoch 50: val_accuracy = 0.88 → Model kaydedilmedi
→ En iyi model (Epoch 5, accuracy=0.90) kaydedildi: best_model.h5
```

### Neden Önemli?

- ✅ **En İyi Modeli Koruma**: Eğitim sırasında en iyi performansı gösteren modeli kaydeder
- ✅ **Güvenlik**: Eğitim kesilirse en iyi model zaten kaydedilmiş olur
- ✅ **Otomatik Kayıt**: Manuel müdahale gerektirmez
- ✅ **Model Karşılaştırma**: Farklı epoch'lardaki modelleri karşılaştırabilirsiniz

### Parametreler:

- **filepath**: Modelin kaydedileceği dosya yolu
- **monitor**: İzlenecek metrik (`'val_accuracy'`, `'val_loss'`, vb.)
- **save_best_only**: Sadece en iyi modeli kaydet (True/False)
- **save_weights_only**: Sadece ağırlıkları kaydet (True/False)
- **verbose**: Bilgilendirme mesajları

### Model Kaydetme Seçenekleri:

```python
# Sadece en iyi modeli kaydet (önerilen)
ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Her epoch'ta kaydet (disk alanı kullanır)
ModelCheckpoint('model_epoch_{epoch}.h5', save_freq='epoch')

# Sadece ağırlıkları kaydet (daha küçük dosya)
ModelCheckpoint('weights.h5', save_weights_only=True)
```

---

## 🔄 Callback'lerin Birlikte Çalışması

Bu üç callback birlikte çalışarak model eğitimini optimize eder:

```
Epoch 1-5:   Model eğitiliyor...
             → ModelCheckpoint: En iyi model kaydediliyor
             
Epoch 6-10:  Validasyon loss iyileşmiyor...
             → ReduceLROnPlateau: LR azaltıldı (0.001 → 0.0005)
             → ModelCheckpoint: Yeni en iyi model kaydediliyor
             
Epoch 11-20: Hala iyileşme yok...
             → Early Stopping: Patience doldu (10/10)
             → Eğitim durduruldu!
             → En iyi model (Epoch 5) geri yüklendi
             → best_model.h5 dosyası hazır
```

---

## 📊 Görsel Örnek

```
Epoch    | val_loss | val_acc | LR      | Action
---------|----------|---------|---------|------------------
1        | 0.50     | 0.85    | 0.001   | ✓ Model kaydedildi
2        | 0.40     | 0.87    | 0.001   | ✓ Model kaydedildi
3        | 0.35     | 0.89    | 0.001   | ✓ Model kaydedildi
4        | 0.36     | 0.88    | 0.001   | ✗ (patience başladı)
5        | 0.37     | 0.87    | 0.001   | ✗ (patience: 1/5)
6        | 0.38     | 0.86    | 0.001   | ✗ (patience: 2/5)
7        | 0.39     | 0.85    | 0.001   | ✗ (patience: 3/5)
8        | 0.40     | 0.84    | 0.001   | ✗ (patience: 4/5)
9        | 0.41     | 0.83    | 0.0005  | → LR azaltıldı!
10       | 0.35     | 0.89    | 0.0005  | ✓ Model kaydedildi
11       | 0.36     | 0.88    | 0.0005  | ✗ (patience: 1/10)
...
20       | 0.45     | 0.80    | 0.0005  | ✗ (patience: 10/10)
         |          |         |         | → Eğitim durduruldu!
         |          |         |         | → En iyi model (Epoch 10) yüklendi
```

---

## ⚙️ Parametre Önerileri

### Hızlı Test İçin:
```python
patience = 3              # Daha hızlı durdurma
reduce_lr_patience = 2    # Daha hızlı LR azaltma
```

### Dikkatli Eğitim İçin:
```python
patience = 15             # Daha uzun bekleme
reduce_lr_patience = 7    # Daha uzun LR bekleme
```

### Yüksek Performans İçin:
```python
patience = 10             # Dengeli
reduce_lr_patience = 5    # Dengeli
factor = 0.5              # LR'yi yarıya indir
min_lr = 0.00001          # Çok küçük minimum LR
```

---

## 💡 İpuçları

1. **Early Stopping Patience**: 
   - Küçük veri setleri için: 5-10
   - Büyük veri setleri için: 10-20

2. **Reduce LR Patience**:
   - Early Stopping'den daha küçük olmalı (örn: 5 vs 10)
   - Böylece önce LR azalır, sonra eğitim durur

3. **Monitor Metrikleri**:
   - `val_loss`: Loss azalmasını izler (düşük = iyi)
   - `val_accuracy`: Accuracy artışını izler (yüksek = iyi)
   - Hangi metrik kullanılmalı? → Genellikle `val_loss` daha güvenilir

4. **Model Checkpoint**:
   - Her zaman `save_best_only=True` kullanın (disk alanı tasarrufu)
   - `monitor='val_accuracy'` veya `monitor='val_loss'` kullanın

---

## 🎯 Özet

| Callback | Ne Yapar? | Ne Zaman Kullanılır? |
|----------|-----------|---------------------|
| **Early Stopping** | Eğitimi durdurur | Overfitting başladığında |
| **Reduce LR** | Öğrenme oranını azaltır | Model yakınsamaya yaklaştığında |
| **Model Checkpoint** | En iyi modeli kaydeder | Her zaman (güvenlik için) |

---

## 📝 Sonuç

Callback'ler, model eğitimini **otomatik olarak optimize eden** araçlardır. Manuel müdahale gerektirmeden:
- ✅ En iyi modeli bulur ve kaydeder
- ✅ Overfitting'i önler
- ✅ Öğrenme oranını optimize eder
- ✅ Zaman ve kaynak tasarrufu sağlar

Bu yüzden her model eğitiminde mutlaka kullanılmalıdırlar! 🚀

