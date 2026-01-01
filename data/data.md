# VERİ SETİ BİLGİLERİ

## 📊 Veri Seti: Digit Recognizer

Bu proje, Kaggle'da bulunan **Digit Recognizer** yarışmasının veri setini kullanmaktadır.

### 🔗 Veri Seti Linki

**Kaggle Yarışma Sayfası:** https://www.kaggle.com/competitions/digit-recognizer

### 📥 Veri Setini İndirme

1. **Kaggle Hesabı Gerekli:**
   - Kaggle hesabınız yoksa [kaggle.com](https://www.kaggle.com) adresinden ücretsiz hesap oluşturun

2. **Veri Setini İndirme:**
   - Yukarıdaki linke tıklayın
   - Yarışma sayfasında "Data" sekmesine gidin
   - Aşağıdaki dosyaları indirin:
     - `train.csv` - Eğitim verisi (label + 784 piksel)
     - `test.csv` - Test verisi (784 piksel)
     - `sample_submission.csv` - Submission formatı örneği

3. **Dosyaları Yerleştirme:**
   - İndirdiğiniz dosyaları `data/` klasörüne kopyalayın
   - Klasör yapısı şöyle olmalı:
     ```
     data/
     ├── train.csv
     ├── test.csv
     └── sample_submission.csv
     ```

### 📋 Veri Seti Özellikleri

- **Eğitim Verisi (train.csv):**
  - Satır sayısı: 42,000
  - Sütunlar: `label` + 784 piksel (pixel0, pixel1, ..., pixel783)
  - Format: CSV
  - Her satır bir 28x28 görüntüyü temsil eder

- **Test Verisi (test.csv):**
  - Satır sayısı: 28,000
  - Sütunlar: 784 piksel (pixel0, pixel1, ..., pixel783)
  - Format: CSV
  - Etiket yok (tahmin yapmamız gerekiyor)

- **Görüntü Özellikleri:**
  - Boyut: 28x28 piksel
  - Renk: Gri tonlamalı (0-255 arası değerler)
  - Format: Düzleştirilmiş (784 piksel tek satırda)

### 🎯 Yarışma Amacı

Bu yarışmada, test verisindeki 28,000 görüntünün her biri için rakam tahmini (0-9) yapmanız gerekmektedir.

### 📝 Submission Formatı

Submission dosyası şu formatta olmalıdır:
```csv
ImageId,Label
1,3
2,7
3,0
...
28000,9
```

### ⚠️ Önemli Notlar

- Veri seti Kaggle'dan indirilmelidir (bu repo'da veri dosyaları bulunmamaktadır)
- Kaggle API kullanarak da indirebilirsiniz:
  ```bash
  kaggle competitions download -c digit-recognizer
  ```
- Veri seti lisansı: Kaggle yarışma kurallarına tabidir

### 🔧 Alternatif İndirme Yöntemleri

#### Kaggle API ile İndirme:

1. **Kaggle API Token Oluşturma:**
   - Kaggle hesabınızda Settings → API → "Create New Token"
   - `kaggle.json` dosyasını `~/.kaggle/` klasörüne kaydedin

2. **Komut Satırından İndirme:**
   ```bash
   pip install kaggle
   kaggle competitions download -c digit-recognizer
   unzip digit-recognizer.zip -d data/
   ```

#### Manuel İndirme:

1. Kaggle yarışma sayfasına gidin
2. "Data" sekmesine tıklayın
3. Her dosyanın yanındaki "Download" butonuna tıklayın
4. Dosyaları `data/` klasörüne kopyalayın

### 📚 Ek Kaynaklar

- **Yarışma Sayfası:** https://www.kaggle.com/competitions/digit-recognizer
- **Kernel'ler (Örnek Çözümler):** Yarışma sayfasında "Code" sekmesinden örnek çözümlere bakabilirsiniz
- **Forum:** Sorularınız için "Discussion" sekmesini kullanabilirsiniz

### ✅ Veri Seti Kontrolü

Dosyaları indirdikten sonra, aşağıdaki komutla kontrol edebilirsiniz:

```python
import pandas as pd

# Eğitim verisi kontrolü
train = pd.read_csv("data/train.csv")
print(f"Eğitim verisi: {train.shape}")  # (42000, 785) olmalı

# Test verisi kontrolü
test = pd.read_csv("data/test.csv")
print(f"Test verisi: {test.shape}")  # (28000, 784) olmalı
```

---

**Not:** Bu veri seti, MNIST veri setinin bir varyasyonudur ve makine öğrenmesi eğitimi için yaygın olarak kullanılmaktadır.
