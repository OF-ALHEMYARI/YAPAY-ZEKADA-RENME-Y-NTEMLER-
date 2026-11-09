import numpy as np
from src.attention import scaled_dot_product_attention

def run_experiment():
    # Sonuçların tekrarlanabilir olması için rastgelelik tohumu (seed) sabitlenir
    np.random.seed(42)
    
    # 1. Sentetik Veri Hazırlığı: 6 Token, her biri 8 boyutlu (d_model = 8)
    num_tokens = 6
    d_model = 8
    d_k = 4  # Query ve Key boyutları
    d_v = 4  # Value boyutu
    
    # Girdi matrisi X (6, 8)
    X = np.random.randn(num_tokens, d_model)
    
    # 2. Q, K, V projeksiyonları için rastgele ağırlık matrisleri
    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_k)
    
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    
    print("--- Temel Matris Boyutları ---")
    print(f"Girdi X boyutu: {X.shape}")
    print(f"Q boyutu: {Q.shape}, K boyutu: {K.shape}, V boyutu: {V.shape}\n")
    
    # 3. Ölçeklendirme (Scaling) ile Attention Hesaplama
    output_scaled, weights_scaled = scaled_dot_product_attention(Q, K, V, scale=True)
    
    print("--- Ölçeklendirilmiş Self-Attention Sonuçları ---")
    print(f"Çıktı (Output) boyutu: {output_scaled.shape}")
    print(f"Dikkat Ağırlıkları (Attention Weights) matrisi boyutu: {weights_scaled.shape}")
    print("Her satırdaki dikkat ağırlıklarının toplamı (yaklaşık 1 olmalıdır):")
    print(np.sum(weights_scaled, axis=-1))

if __name__ == "__main__":
    run_experiment()
