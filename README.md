# Convolutional Neural Network (CNN) - Pengenalan Lengkap

Berikut adalah penjelasan mendalam mengenai Convolutional Neural Network (CNN) dan implementasinya dalam proyek ini.

## Daftar Isi
- [Apa itu CNN?](#apa-itu-convolutional-neural-network-cnn)
- [Komponen Utama](#komponen-utama-cnn)
- [Cara Kerja CNN](#cara-kerja-cnn-alur-kerja)
- [Library yang Digunakan](#library-yang-digunakan)
- [Arsitektur Model](#jumlah-layer-yang-digunakan)
- [Instalasi](#cara-instalasi)
- [Proses Eksekusi](#proses-yang-terjadi-di-balik-layar)
- [Hasil](#hasil-akhir)

---

## Apa itu Convolutional Neural Network (CNN)?

Convolutional Neural Network (CNN) adalah jenis jaringan saraf tiruan (neural network) yang sangat efektif dalam pemrosesan dan analisis data visual. CNN dirancang khusus untuk mengenali pola spasial dalam data, membuatnya ideal untuk tugas-tugas seperti pengenalan gambar, klasifikasi video, dan pemrosesan bahasa alami.

### Komponen Utama CNN
CNN terdiri dari beberapa lapisan (layer) yang bekerja bersama untuk mengekstrak fitur dan membuat prediksi:

1.  **Convolutional Layer (Lapisan Konvolusi)**
    *   Merupakan inti dari CNN. Lapisan ini melakukan operasi konvolusi pada input.
    *   **Filter (Kernel)**: Matriks kecil (misalnya, 3x3 atau 5x5) yang bergeser di seluruh gambar input. Setiap filter dirancang untuk mendeteksi fitur tertentu seperti tepi, tekstur, atau bentuk.
    *   **Operasi Konvolusi**: Filter dikalikan elemen demi elemen dengan bagian gambar yang ditutupi, dan hasilnya dijumlahkan untuk menghasilkan satu nilai dalam peta fitur output. Proses ini diulangi di seluruh gambar.
    *   **Peta Fitur (Feature Map)**: Output dari operasi konvolusi. Setiap filter menghasilkan peta fitur terpisah yang menunjukkan di mana fitur yang terdeteksi muncul dalam gambar input.
    *   **Padding**: Menambahkan piksel bernilai nol di sekitar tepi gambar untuk mengontrol ukuran output.
    *   **Stride**: Jumlah langkah filter bergerak di setiap pergeseran. Stride yang lebih besar menghasilkan peta fitur output yang lebih kecil.

2.  **Activation Function (Fungsi Aktivasi)**

    Setelah konvolusi, fungsi aktivasi non-linear diterapkan ke peta fitur.
    *   **ReLU (Rectified Linear Unit)**: Fungsi ini mengubah semua nilai negatif menjadi nol, sementara nilai positif tetap tidak berubah: `f(x) = max(0, x)`. Ini memperkenalkan non-linearitas, memungkinkan jaringan untuk mempelajari pola yang kompleks.

3.  **Pooling Layer (Lapisan Pooling)**

    Lapisan ini mengurangi dimensi spasial (lebar dan tinggi) dari peta fitur, mengurangi jumlah parameter dan komputasi, serta membantu mencegah overfitting.
    *   **Max Pooling**: Jenis yang paling umum; memilih nilai maksimum dari setiap jendela kecil (misalnya, 2x2) dalam peta fitur.
    *   **Average Pooling**: Mengambil nilai rata-rata alih-alih maksimum.

4.  **Fully Connected Layer (Dense Layer) / Lapisan Terhubung Penuh**

    Setelah beberapa lapisan konvolusi dan pooling, peta fitur yang diekstraksi diratakan menjadi vektor panjang tunggal.
    *   Vektor ini dimasukkan ke dalam satu atau lebih lapisan terhubung penuh, mirip dengan jaringan saraf tradisional.
    *   Lapisan-lapisan ini bertanggung jawab untuk melakukan klasifikasi atau regresi berdasarkan fitur tingkat tinggi yang diekstraksi oleh lapisan sebelumnya.
    *   Lapisan output terakhir biasanya menggunakan Softmax (untuk klasifikasi multi-kelas) atau fungsi aktivasi Sigmoid (untuk klasifikasi biner).

### Cara Kerja CNN (Alur Kerja)

Misalkan Anda ingin melatih CNN untuk mengenali apakah sebuah gambar berisi kucing atau anjing:

1.  **Gambar Input**: Sebuah gambar (misalnya, 224x224 piksel dengan 3 saluran RGB) dimasukkan ke dalam CNN.
2.  **Ekstraksi Fitur (Lapisan Konvolusi & Pooling)**:
    *   **Lapisan Konvolusi Pertama**: Filter kecil (misalnya, 3x3) meluncur di atas gambar untuk mendeteksi fitur yang sangat dasar seperti tepi horizontal, vertikal, atau diagonal. Outputnya adalah sekumpulan peta fitur.
    *   **Fungsi Aktivasi (ReLU)**: Non-linearitas diterapkan ke peta fitur.
    *   **Lapisan Pooling Pertama**: Mengurangi ukuran peta fitur (misalnya, dari 224x224 menjadi 112x112) sambil mempertahankan fitur yang paling penting.
    *   Proses ini (Konvolusi → Aktivasi → Pooling) diulangi beberapa kali. Setiap lapisan konvolusi yang lebih dalam mempelajari fitur yang lebih kompleks dengan menggabungkan pola yang terdeteksi oleh lapisan sebelumnya—seperti mendeteksi mata, hidung, atau telinga.
    *   Semakin dalam lapisan, semakin abstrak dan kompleks fitur yang diekstraksi.
3.  **Perataan (Flattening)**: Setelah melewati beberapa lapisan konvolusi dan pooling, peta fitur akhir diratakan menjadi vektor panjang tunggal.
4.  **Klasifikasi (Lapisan Terhubung Penuh)**:
    *   Vektor fitur ini dimasukkan ke dalam satu atau lebih lapisan terhubung penuh. Lapisan-lapisan ini belajar untuk menggabungkan fitur yang diekstraksi dan membuat keputusan akhir.
    *   **Lapisan Output**: Lapisan terakhir (misalnya, dengan 2 neuron untuk "kucing" dan "anjing" dan aktivasi softmax) mengeluarkan probabilitas gambar menjadi kucing atau anjing.
5.  **Pelatihan**: Selama pelatihan, jaringan diperlihatkan banyak contoh kucing dan anjing yang diberi label. Menggunakan algoritma seperti backpropagation dan gradient descent, bobot filter dan neuron disesuaikan untuk meminimalkan kesalahan prediksi.

## Library yang Digunakan

*   **TensorFlow**: TensorFlow adalah library sumber terbuka yang dikembangkan oleh Google untuk machine learning dan deep learning. TensorFlow menyediakan berbagai alat untuk membangun dan melatih model machine learning.

*   **Keras**: Keras adalah API tingkat tinggi untuk membangun dan melatih model deep learning. Keras berjalan di atas TensorFlow dan menyediakan cara yang lebih sederhana dan intuitif untuk mendefinisikan model.

*   **NumPy**: NumPy adalah library Python untuk komputasi numerik. NumPy menyediakan dukungan untuk array multidimensi dan berbagai fungsi matematika.

*   **Matplotlib**: Matplotlib adalah library Python untuk membuat visualisasi data. Matplotlib dapat digunakan untuk membuat berbagai jenis grafik, seperti grafik garis, grafik batang, dan scatter plot.

## Jumlah Layer yang Digunakan

Model CNN yang digunakan dalam contoh ini memiliki lapisan-lapisan berikut:

1.  Convolutional Layer (Conv2D): 3 lapisan
2.  Max Pooling Layer (MaxPooling2D): 2 lapisan
3.  Flatten Layer: 1 lapisan
4.  Dense Layer (Fully Connected): 2 lapisan

Total: 8 lapisan

## Instalasi

### Prerequisites
- Python 3.7 atau lebih tinggi
- pip (Python Package Manager)

### Setup dengan pip

Untuk menjalankan kode ini, Anda perlu menginstal library yang diperlukan:

```bash
pip install tensorflow keras numpy matplotlib
```

Atau gunakan requirements.txt jika tersedia:

```bash
pip install -r requirements.txt
```

### Verifikasi Instalasi

Untuk memverifikasi bahwa semua library telah terinstal dengan benar:

```python
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Setup berhasil!")
```

## Proses yang Terjadi di Balik Layar

### Alur Eksekusi Umum

1.  **Pemuatan Data**: Dataset Fashion MNIST dimuat dan dibagi menjadi data pelatihan dan pengujian.
2.  **Preprocessing**: 
    - Nilai piksel dinormalisasi ke rentang 0-1
    - Gambar-gambar diubah bentuknya agar sesuai dengan format input lapisan konvolusi
3.  **Pembuatan Model**: Model CNN dibuat menggunakan API Keras Sequential dengan arsitektur layer yang telah ditentukan
4.  **Kompilasi Model**: 
    - Optimizer: Adam
    - Loss Function: SparseCategoricalCrossentropy
    - Metrics: Accuracy
5.  **Pelatihan Model**: Model menyesuaikan bobotnya melalui backpropagation untuk meminimalkan loss
6.  **Evaluasi Model**: Model dievaluasi menggunakan data pengujian untuk mengukur performa pada data yang tidak terlihat
7.  **Visualisasi Hasil**: Akurasi dan kerugian divisualisasikan menggunakan Matplotlib
8.  **Membuat Prediksi**: Model terlatih melakukan prediksi pada gambar pengujian
9.  **Visualisasi Prediksi**: Hasil prediksi divisualisasikan dan dibandingkan dengan label sebenarnya

---

## Cara Menjalankan Project

### Dari Jupyter Notebook

1. Buka file `CNNexample.ipynb` di Jupyter atau VS Code
2. Jalankan semua cell dengan urutan dari atas ke bawah
3. Amati output dan visualisasi yang dihasilkan

### Dari Script Python

Jika Anda ingin menjalankan dari script Python:

```bash
python CNNexample.py
```

---

---

## Struktur Folder

```
CNN-Convolutional-Neural-Network-/
├── CNNexample.ipynb          # Notebook utama dengan semua kode dan penjelasan
├── Convolution/              # Folder yang berisi file-file konvolusi tambahan
├── README.md                 # Dokumentasi proyek (file ini)
└── [file lainnya]
```

### Deskripsi File

- **CNNexample.ipynb**: Notebook Jupyter yang berisi implementasi lengkap CNN dengan penjelasan step-by-step
- **Convolution/**: Folder yang mungkin berisi demonstrasi, helper functions, atau contoh konvolusi manual

---

## Hasil Akhir

Setelah menjalankan kode, Anda akan mendapatkan:

- **Ringkasan Arsitektur Model CNN** - Print dari model dengan jumlah parameter total
- **Metrik Performa** - Akurasi dan kerugian pada data pelatihan dan pengujian
- **Visualisasi Grafik** - Plot akurasi dan kerugian selama pelatihan untuk analisis performa
- **Prediksi Sampel** - Hasil prediksi pada beberapa gambar test dengan confidence score

---

## Tips & Troubleshooting

### Tips untuk Meningkatkan Akurasi

1. **Tambah Epochs**: Naikkan jumlah epoch training untuk pembelajaran yang lebih dalam
2. **Augmentasi Data**: Gunakan data augmentation untuk menambah variasi dataset
3. **Hyperparameter Tuning**: Eksperimen dengan learning rate, batch size, dan regularization
4. **Arsitektur Lebih Kompleks**: Tambahkan lebih banyak layer atau filter untuk model yang lebih powerful

### Troubleshooting Umum

- **Out of Memory Error**: Kurangi batch size atau ukuran filter
- **Model Overfit**: Tambahkan Dropout layer atau L2 regularization
- **Performa Lambat**: Gunakan GPU untuk akselerasi (pastikan CUDA terinstall dengan baik)

---

## Referensi & Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Stanford CS231N - Convolutional Neural Networks](https://cs231n.github.io/)

---

## Catatan Tambahan

- Proyek ini menggunakan **Fashion MNIST Dataset** yang merupakan dataset fashion items grey-scale images
- Model yang dibangun relatif sederhana untuk keperluan pembelajaran dan dapat dioptimasi lebih lanjut
- Silakan eksplorasi folder `Convolution/` untuk melihat implementasi detail dari operasi konvolusi
- Dengan pemahaman yang lebih mendalam tentang CNN, Anda dapat bereksperimen dengan arsitektur yang berbeda, hyperparameter yang berbeda, dan dataset yang berbeda

---

