import time
import numpy as np
from src.attention import scaled_dot_product_attention

def run_complexity_experiment():
    """
    Self-Attention mekanizmasının dizi uzunluğuna (Sequence Length - T) 
    göre hesaplama karmaşıklığını (O(T^2)) ölçen deney.
    """
    np.random.seed(42)
    
    # Test edilecek farklı dizi uzunlukları (Token sayıları)
    sequence_lengths = [8, 16, 32, 64, 128, 256]
    d_model = 8
    d_k = 4
    
    print("--- Zaman ve Bellek Karmaşıklığı Analizi O(T^2) ---")
    print(f"{'Dizi Uzunluğu (T)':<18} | {'Matris Eleman Sayısı (T^2)':<25} | {'Süre (saniye)'}")
    print("-" * 70)
    
    for T in sequence_lengths:
        X = np.random.randn(T, d_model)
        W_q = np.random.randn(d_model, d_k)
        W_k = np.random.randn(d_model, d_k)
        W_v = np.random.randn(d_model, d_k)
        
        Q = np.dot(X, W_q)
        K = np.dot(X, W_k)
        V = np.dot(X, W_v)
        
        # Süre ölçümü başlangıcı
        start_time = time.time()
        _, weights = scaled_dot_product_attention(Q, K, V, scale=True)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        num_elements = T * T
        
        print(f"{T:<18} | {num_elements:<25} | {elapsed_time:.6f}")

if __name__ == "__main__":
    run_complexity_experiment()
