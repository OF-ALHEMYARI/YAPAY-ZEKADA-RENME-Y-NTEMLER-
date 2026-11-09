import numpy as np

def stable_softmax(scores: np.ndarray) -> np.ndarray:
    """
    Sayısal kararlılığa sahip (numerically stable) Softmax fonksiyonu.
    Son eksen (axis) üzerinden hesaplama yapar.
    """
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: bool = True):
    """
    Ölçeklendirilmiş Nokta Çarpım Dikkat Mekanizması (Scaled Dot-Product Attention)
    """
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T)
    
    if scale:
        scores = scores / np.sqrt(d_k)
        
    attention_weights = stable_softmax(scores)
    output = np.dot(attention_weights, V)
    
    return output, attention_weights
