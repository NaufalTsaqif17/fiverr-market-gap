# Fiverr Market Gap Analysis

Proyek Ujian Akhir Semester Pembelajaran Mesin untuk menganalisis
market gap layanan Data dan AI pada platform Fiverr.

## Problem Statement

Calon freelancer sering kesulitan menentukan layanan yang layak
ditawarkan karena setiap niche mempunyai tingkat kompetisi, harga,
dan potensi permintaan yang berbeda.

Proyek ini menggunakan NLP dan Machine Learning untuk mengelompokkan
gig, memprediksi potensi permintaan, dan menghasilkan Market Gap Score.

## Dataset

Dataset berisi 1.259 gig Fiverr dengan atribut:

- Judul gig
- URL gig
- Rating
- Jumlah review
- Seller level
- Harga dalam PKR

## Metodologi

1. Data cleaning dan preprocessing
2. Exploratory Data Analysis
3. TF-IDF text feature extraction
4. Ridge Regression
5. XGBoost Regressor
6. Hyperparameter tuning
7. SHAP model interpretation
8. K-Means clustering
9. Market Gap Score
10. Streamlit deployment

## Struktur Repository

```text
app/
data/
models/
notebooks/
reports/
src/
