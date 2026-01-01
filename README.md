# Modüler CNN Modeli - Dosya Yapısı

Bu proje, CNN modeli için modüler bir yapı sunar. Her modül ayrı bir dosyada bulunur ve `main.py`'den çağrılır.

## 📁 Dosya Yapısı

```
VeriHazirlamaGoruntu/
├── main.py                          # Ana program - Tüm modülleri çağırır
├── config.py                        # Yapılandırma dosyası (tüm ayarlar)
├── modules/                         # Modül klasörü
│   ├── __init__.py                 # Paket başlatma dosyası
│   ├── data_loader.py              # Modül 2: Veri yükleme ve ön işleme
│   ├── data_visualization.py       # Modül 3: Veri görselleştirme
│   ├── data_augmentation_module.py # Modül 4: Veri artırma
│   ├── model_builder.py            # Modül 5: CNN modeli oluşturma
│   ├── callbacks.py                # Modül 6: Callback'ler
│   ├── model_trainer.py            # Modül 7: Model eğitimi
│   ├── training_visualizer.py      # Modül 8: Eğitim geçmişi görselleştirme
│   ├── model_evaluator.py          # Modül 9: Model değerlendirme
│   └── predictor.py                # Modül 10: Tahmin ve submission
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── README_MODULES.md               # Bu dosya
```

## 🚀 Kullanım

### Basit Kullanım

```bash
python main.py
```

Bu komut, `config.py` dosyasındaki ayarları kullanarak tüm pipeline'ı çalıştırır.

### Özelleştirilmiş Kullanım

`main.py` dosyasını düzenleyerek parametreleri özelleştirebilirsiniz:

```python
model, history, results, predictions = run_complete_pipeline(
    model_type='deep',        # 'simple', 'standard', 'deep'
    epochs=100,
    batch_size=64,
    use_augmentation=True,
    save_model_path='my_model.h5',
    submission_path='my_submission.csv'
)
```

### Yapılandırma Dosyası (config.py)

Tüm ayarlar `config.py` dosyasında toplanmıştır:

```python
# Model ayarları
MODEL_TYPE = 'standard'  # 'simple', 'standard', 'deep'
EPOCHS = 50
BATCH_SIZE = 32

# Veri artırma
USE_AUGMENTATION = True
ROTATION_RANGE = 5
WIDTH_SHIFT_RANGE = 0.1

# ... diğer ayarlar
```

## 📚 Modüller

### Modül 2: data_loader.py
**Fonksiyon:** `load_and_preprocess_data()`
- CSV dosyalarından veri yükleme
- Normalizasyon
- Reshape işlemi
- One-Hot Encoding
- Train/Validation split

### Modül 3: data_visualization.py
**Fonksiyon:** `visualize_data()`
- Sınıf dağılımı grafiği
- Örnek görüntüler

### Modül 4: data_augmentation_module.py
**Fonksiyonlar:**
- `create_data_generator()` - ImageDataGenerator oluşturur
- `create_train_generator()` - Eğitim generator'ı oluşturur

### Modül 5: model_builder.py
**Fonksiyon:** `create_cnn_model()`
- 3 model tipi: simple, standard, deep
- Model derleme

### Modül 6: callbacks.py
**Fonksiyon:** `create_callbacks()`
- Early Stopping
- Learning Rate Reduction
- Model Checkpoint

### Modül 7: model_trainer.py
**Fonksiyon:** `train_model()`
- Veri artırma ile/olmadan eğitim
- Otomatik steps_per_epoch hesaplama

### Modül 8: training_visualizer.py
**Fonksiyon:** `plot_training_history()`
- Accuracy grafiği
- Loss grafiği

### Modül 9: model_evaluator.py
**Fonksiyon:** `evaluate_model()`
- Accuracy ve Loss metrikleri
- Confusion Matrix
- Classification Report

### Modül 10: predictor.py
**Fonksiyon:** `make_predictions()`
- Test verisi üzerinde tahmin
- CSV submission dosyası oluşturma

## 🔧 Modül Kullanımı (Bağımsız)

Her modülü bağımsız olarak da kullanabilirsiniz:

```python
# Sadece veri yükleme
from modules.data_loader import load_and_preprocess_data
X_train, X_val, Y_train, Y_val, X_test = load_and_preprocess_data()

# Sadece model oluşturma
from modules.model_builder import create_cnn_model
model = create_cnn_model(model_type='standard')

# Sadece değerlendirme
from modules.model_evaluator import evaluate_model
results = evaluate_model(model, X_val, Y_val)
```

## ⚙️ Yapılandırma Seçenekleri

### Model Tipleri

1. **Simple**: Hızlı, basit model
   ```python
   MODEL_TYPE = 'simple'
   ```

2. **Standard**: Dengeli model (önerilen)
   ```python
   MODEL_TYPE = 'standard'
   ```

3. **Deep**: Derin, yüksek performanslı model
   ```python
   MODEL_TYPE = 'deep'
   ```

### Veri Artırma

```python
USE_AUGMENTATION = True
ROTATION_RANGE = 5          # ±5 derece
WIDTH_SHIFT_RANGE = 0.1      # %10 yatay kaydırma
HEIGHT_SHIFT_RANGE = 0.1     # %10 dikey kaydırma
ZOOM_RANGE = 0.1             # %10 yakınlaştırma
```

### Eğitim Parametreleri

```python
EPOCHS = 50                 # Epoch sayısı
BATCH_SIZE = 32              # Batch boyutu
VALIDATION_SPLIT = 0.1       # %10 doğrulama seti
RANDOM_STATE = 13            # Rastgelelik seed'i
```

### Callback Ayarları

```python
EARLY_STOPPING_PATIENCE = 10      # Early stopping sabır değeri
REDUCE_LR_PATIENCE = 5            # LR reduction sabır değeri
```

## 📊 Çıktılar

Program çalıştırıldığında:

1. **Veri görselleştirmeleri** (grafikler)
2. **Model özeti** (katmanlar ve parametreler)
3. **Eğitim geçmişi grafikleri** (accuracy ve loss)
4. **Confusion Matrix**
5. **Classification Report**
6. **Model dosyası** (`best_model.h5`)
7. **Submission dosyası** (`submission.csv`)

## 💡 İpuçları

1. **İlk Deneme**: `MODEL_TYPE = 'simple'` ile başlayın
2. **En İyi Performans**: `MODEL_TYPE = 'standard'` + `USE_AUGMENTATION = True`
3. **Yüksek Doğruluk**: `MODEL_TYPE = 'deep'` + daha fazla epoch
4. **Hızlı Test**: `EPOCHS = 5` ile hızlı test yapın
5. **GPU Kullanımı**: GPU varsa otomatik kullanılır

## 🐛 Sorun Giderme

### Import Hatası
```python
# Modüllerin doğru import edildiğinden emin olun
from modules.data_loader import load_and_preprocess_data
```

### Dosya Yolu Hatası
```python
# config.py'de dosya yollarını kontrol edin
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
```

### Memory Hatası
```python
# config.py'de batch_size'ı azaltın
BATCH_SIZE = 16  # veya 8
```

## 📝 Notlar

- Tüm modüller bağımsız çalışabilir
- `config.py` dosyasından tüm ayarları yönetebilirsiniz
- Model otomatik olarak en iyi ağırlıklarla kaydedilir
- Early Stopping ile gereksiz eğitim önlenir
- Submission dosyası otomatik oluşturulur

## 🔄 Güncelleme

Yeni bir modül eklemek için:

1. `modules/` klasörüne yeni dosya ekleyin
2. `main.py`'de import edin
3. `run_complete_pipeline()` fonksiyonunda kullanın

## 📧 Destek

Sorularınız için kod içindeki yorumları inceleyin.

