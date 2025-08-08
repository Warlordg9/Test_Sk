import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
import random

class RocketTransformer(BaseEstimator, TransformerMixin):
    """Реализация ROCKET для преобразования временных рядов"""
    
    def __init__(self, n_kernels=10000, kernel_types=['normal', 'binary', 'ternary'], 
                 kernel_lengths=[7, 9, 11], random_state=None):
        """
        Инициализация трансформера
        
        Параметры:
            n_kernels: количество случайных ядер
            kernel_types: типы инициализации ['normal', 'binary', 'ternary']
            kernel_lengths: возможные длины ядер
            random_state: seed для воспроизводимости
        """
        self.n_kernels = n_kernels
        self.kernel_types = kernel_types
        self.kernel_lengths = kernel_lengths
        self.random_state = random_state
        self.kernels = []
        
    def _generate_kernels(self, input_length):
        """Генерация случайных сверточных ядер"""
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            
        kernels = []
        for _ in range(self.n_kernels):
            # Случайные параметры ядра
            kernel_type = random.choice(self.kernel_types)
            kernel_length = random.choice(self.kernel_lengths)
            
            # Генерация весов
            if kernel_type == 'normal':
                weights = np.random.normal(0, 1, kernel_length)
            elif kernel_type == 'binary':
                weights = np.random.choice([-1, 1], kernel_length)
            elif kernel_type == 'ternary':
                weights = np.random.choice([-1, 0, 1], kernel_length, p=[1/3, 1/3, 1/3])
            
            # Нормализация весов
            weights = weights - np.mean(weights)
            if random.random() > 0.5:
                weights = -weights
                
            # Параметры свертки
            max_dilation = max(1, (input_length - 1) // (kernel_length - 1))
            dilation = 2 ** random.uniform(0, np.log2(max_dilation))
            padding = random.randint(0, int((kernel_length - 1) * dilation))
            bias = random.uniform(-1, 1)
            
            kernels.append({
                'weights': weights.astype(np.float32),
                'bias': bias,
                'dilation': int(dilation),
                'padding': padding
            })
            
        return kernels
    
    def _apply_kernel(self, x, kernel):
        """Применение одного ядра к временному ряду"""
        weights = kernel['weights']
        bias = kernel['bias']
        dilation = kernel['dilation']
        padding = kernel['padding']
        L = len(weights)
        
        # Применение дилатации и паддинга
        total_padding = (L - 1) * dilation
        padded_x = np.pad(x, (padding, total_padding - padding), 
                          mode='constant', constant_values=0)
        
        # Вычисление свертки
        indices = np.arange(len(x) + padding - total_padding)[:, None] + np.arange(L) * dilation
        segments = padded_x[indices]
        conv_out = np.sum(segments * weights, axis=1) + bias
        
        # Извлечение признаков
        max_val = np.max(conv_out)
        ppv = np.mean(conv_out > 0)
        
        return max_val, ppv
    
    def fit(self, X, y=None):
        """Обучение трансформера (генерация ядер)"""
        _, input_length = X.shape
        self.kernels = self._generate_kernels(input_length)
        return self
    
    def transform(self, X, n_jobs=-1):
        """Преобразование временных рядов в признаки"""
        features = Parallel(n_jobs=n_jobs)(
            delayed(self._transform_single)(x) for x in X
        )
        return np.array(features)
    
    def _transform_single(self, x):
        """Преобразование одного временного ряда"""
        ts_features = np.zeros(2 * len(self.kernels), dtype=np.float32)
        for i, kernel in enumerate(self.kernels):
            max_val, ppv = self._apply_kernel(x, kernel)
            ts_features[2*i] = max_val
            ts_features[2*i+1] = ppv
        return ts_features 
