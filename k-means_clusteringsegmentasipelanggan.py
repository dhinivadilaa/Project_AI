import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Untuk visualisasi yang lebih baik
from sklearn.cluster import KMeans # Algoritma k-Means
from sklearn.preprocessing import StandardScaler # Untuk scaling fitur
import sys # Untuk keluar program jika error

def simple_customer_segmentation_kmeans(data_file_path, num_clusters=5):
    print("--- Memulai Segmentasi Pelanggan Sederhana dengan k-Means ---")

    # 1. Pemuatan Data
    try:
        df = pd.read_csv(data_file_path)
        print(f"Dataset '{data_file_path}' berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File '{data_file_path}' tidak ditemukan. Pastikan ada di direktori yang sama.")
        sys.exit(1)
    except Exception as e:
        print(f"Error saat membaca file CSV: {e}")
        sys.exit(1)

    # 2. Pemilihan Fitur
    # Menggunakan Pendapatan Tahunan dan Skor Pengeluaran
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    # Periksa apakah kolom ada
    if not all(col in df.columns for col in features):
        print(f"Error: Salah satu atau kedua kolom fitur '{features}' tidak ditemukan di dataset.")
        sys.exit(1)

    X = df[features].copy() 

    # 3. Penanganan Data Hilang (jika ada)
    if X.isnull().sum().any():
        X.fillna(X.mean(), inplace=True)
        print("Nilai yang hilang pada fitur diisi dengan rata-rata.")

    # 4. Scaling Fitur
    # Penting agar fitur punya skala yang sama
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Fitur berhasil diskalakan.")

    # 5. Penerapan k-Means Clustering
    print(f"\nMelakukan clustering dengan K = {num_clusters}...")
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42, n_init='auto')
    
    # Melatih model dan menetapkan label cluster ke setiap pelanggan
    df['Cluster'] = kmeans_model.fit_predict(X_scaled)
    print("Clustering selesai. Label cluster ditambahkan.")

    # Mendapatkan pusat cluster (centroids) dalam skala asli untuk visualisasi
    centroids_original_scale = scaler.inverse_transform(kmeans_model.cluster_centers_)
    
    # 6. Visualisasi Hasil Clustering
    print("Menampilkan visualisasi hasil clustering...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=df, 
                    palette='viridis', s=100, alpha=0.8, edgecolor='w')
    
    plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1], 
                s=300, c='red', marker='X', label='Pusat Cluster', edgecolor='black')
    
    plt.title(f'Segmentasi Pelanggan dengan k-Means (K={num_clusters})')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Visualisasi berhasil ditampilkan.")

    # Ringkasan sederhana
    print("\n--- Ringkasan Cluster ---")
    print("Jumlah pelanggan per cluster:")
    print(df['Cluster'].value_counts().sort_index())
    print("\nPusat Cluster (rata-rata Pendapatan dan Pengeluaran untuk setiap kelompok):")
    print(pd.DataFrame(centroids_original_scale, columns=features))

# --- Bagian Utama Program ---
if __name__ == "__main__":
    DATASET_FILE_NAME = 'Mall_Customers.csv' 
    NUM_CLUSTERS_TO_USE = 5 # Anda bisa mengubah nilai K ini

    simple_customer_segmentation_kmeans(DATASET_FILE_NAME, num_clusters=NUM_CLUSTERS_TO_USE)
