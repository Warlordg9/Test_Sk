import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tslearn.datasets import UCR_UEA_datasets
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
import os
from scipy.stats import sem, t

class ROCKET:
    """
    num_kernels (int): Количество случайных ядер (по умолчанию 10,000)
    kernel_types (list): Типы генерируемых ядер ['normal', 'binary', 'ternary']
    random_state (int): Seed для воспроизводимости
    """
    
    def __init__(self, num_kernels=10000, kernel_types=['normal'], random_state=None):
        self.num_kernels = num_kernels
        self.kernel_types = kernel_types
        self.kernels = []
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
    def _generate_kernel(self, input_length):
        """Генерация случайного ядра с дилатацией"""
        # 1. Выбор длины ядра (экспоненциальное распределение)
        max_len = min(128, input_length // 2)
        exp_sample = np.random.exponential(scale=10)
        length = int(min(max_len, max(7, 2 * np.floor(exp_sample / 2) + 1)))
        
        # 2. Расчет дилатации
        max_dilation = np.log2((input_length - 1) / (length - 1)) if length > 1 else 0
        if max_dilation <= 0:
            dilation = 1
        else:
            x = np.random.uniform(0, max_dilation)
            dilation = int(2 ** x)
        
        # 3. Выбор типа весов
        weights_type = np.random.choice(self.kernel_types)
        
        # 4. Генерация весов
        if weights_type == 'normal':
            weights = np.random.normal(0, 1, size=length)
        elif weights_type == 'binary':
            weights = np.random.choice([-1, 1], size=length)
        else:  # ternary
            weights = np.random.choice([-1, 0, 1], size=length, p=[1/6, 2/3, 1/6])
        
        # 5. Нормализация весов
        weights = (weights - np.mean(weights)) / (np.std(weights) + 1e-8)
        
        # 6. Случайное смещение
        bias = np.random.uniform(-1, 1)
        
        return {
            'weights': weights,
            'length': length,
            'dilation': dilation,
            'bias': bias,
            'type': weights_type
        }
    
    def fit(self, X):
        """Генерация случайных ядер на основе обучающих данных"""
        self.kernels = []
        input_length = X.shape[1]
        
        for _ in range(self.num_kernels):
            kernel = self._generate_kernel(input_length)
            self.kernels.append(kernel)
        
        return self
    
    def _apply_kernel(self, series, kernel):
        """Применение одного ядра к временному ряду"""
        length = kernel['length']
        dilation = kernel['dilation']
        weights = kernel['weights']
        bias = kernel['bias']
        
        # Вычисление длины выхода с учетом дилатации
        output_length = series.shape[0] - (length - 1) * dilation
        if output_length < 1:
            # Если короткий возвращаем нули
            return np.zeros(1)
        
        # Дилатированная свертка
        convolutions = np.zeros(output_length)
        for i in range(output_length):
            start = i * dilation
            end = start + (length - 1) * dilation + 1
            segment = series[start:end:dilation]
            convolutions[i] = np.dot(segment, weights) + bias
        
        return convolutions
    
    def transform(self, X):
        """Преобразование  в признаки"""
        features = np.zeros((X.shape[0], len(self.kernels) * 2))
        
        for i, series in enumerate(X):
            series_features = []
            for kernel in self.kernels:
                conv = self._apply_kernel(series, kernel)
                
                # Два признака на ядро: максимум и PPV (proportion of positive values)
                max_val = np.max(conv) if len(conv) > 0 else 0
                ppv = np.mean(conv > 0) if len(conv) > 0 else 0
                
                series_features.extend([max_val, ppv])
            
            features[i] = np.array(series_features)
        
        return features

def load_data(dataset_name='Ham'):
    data_loader = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset_name)
    
    # Обработка различных форматов 
    if isinstance(X_train, tuple):
        X_train, y_train, X_test, y_test = X_train
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    # Преобразование 3D в 2D для одномерных рядов
    if X_train.ndim == 3 and X_train.shape[2] == 1:
        X_train = X_train[:, :, 0]
        X_test = X_test[:, :, 0]
    
    return X_train, y_train, X_test, y_test

def run_experiment(kernel_type, num_runs=10, num_kernels=10000):
    """Эксперимент для одного типа ядер"""
    accuracies = []
    times = []
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs} for {kernel_type} kernels...")
        start_time = time.time()
        
        try:
            X_train, y_train, X_test, y_test = load_data()
            
            # Инициализация ROCKET
            rocket = ROCKET(
                num_kernels=num_kernels,
                kernel_types=[kernel_type],
                random_state=np.random.randint(0, 10000)
            )
            rocket.fit(X_train)
            
            # Преобразование 
            X_train_transformed = rocket.transform(X_train)
            X_test_transformed = rocket.transform(X_test)
            
            # Масштабирование
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_transformed)
            X_test_scaled = scaler.transform(X_test_transformed)
            
            # Обучение классификатора
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier.fit(X_train_scaled, y_train)
            
            # Оценка
            y_pred = classifier.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            
            accuracies.append(acc)
            times.append(time.time() - start_time)
            print(f"  Accuracy: {acc:.4f}, Time: {times[-1]:.2f}s")
        except Exception as e:
            print(f"Error in run {run+1}: {str(e)}")
            accuracies.append(0)
            times.append(0)
    
    return accuracies, times

def calculate_confidence_interval(data, confidence=0.95):
    """Доверительный интервал"""
    n = len(data)
    if n < 2:
        return np.mean(data), 0, 0
    
    mean = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def main():
    parser = argparse.ArgumentParser(description='ROCKET Time Series Classification')
    parser.add_argument('--num_runs', type=int, default=10, 
                        help='Количество запусков для каждого типа ядер')
    parser.add_argument('--num_kernels', type=int, default=10000, 
                        help='Количество генерируемых ядер')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Директория для сохранения результатов')
    parser.add_argument('--dataset', type=str, default='Ham', 
                        help='Название датасета (по умолчанию: Ham)')
    args = parser.parse_args()
    
    # Создание директории для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # Проведение экспериментов для всех типов ядер
    for kernel_type in ['normal', 'binary', 'ternary']:
        print(f"\n{'='*50}")
        print(f"Starting experiments for {kernel_type} kernels")
        print(f"{'='*50}")
        
        accuracies, times = run_experiment(
            kernel_type,
            num_runs=args.num_runs,
            num_kernels=args.num_kernels
        )
        
        # Расчет статистик
        acc_mean, acc_low, acc_high = calculate_confidence_interval(accuracies)
        time_mean = np.mean(times)
        
        results[kernel_type] = {
            'accuracies': accuracies,
            'times': times,
            'accuracy_mean': acc_mean,
            'accuracy_ci': (acc_low, acc_high),
            'time_mean': time_mean
        }
        
        print(f"\n{kernel_type} kernels summary:")
        print(f"  Mean accuracy: {acc_mean:.4f}")
        print(f"  95% CI: ({acc_low:.4f} - {acc_high:.4f})")
        print(f"  Mean time per run: {time_mean:.2f}s")
    
    # Визуализация
    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid")
    data_to_plot = [
        results['normal']['accuracies'],
        results['binary']['accuracies'],
        results['ternary']['accuracies']
    ]
    
    sns.boxplot(data=data_to_plot)
    plt.xticks([0, 1, 2], ['Normal', 'Binary', 'Ternary'])
    plt.ylabel('Accuracy')
    plt.title(f'ROCKET Performance Comparison ({args.dataset} Dataset)')
    plt.savefig(os.path.join(args.output_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # Сохранение результатов в текстовый файл
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("ROCKET EXPERIMENT RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of runs per kernel type: {args.num_runs}\n")
        f.write(f"Number of kernels: {args.num_kernels}\n\n")
        
        f.write("="*50 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*50 + "\n")
        
        for kernel_type, res in results.items():
            f.write(f"\nKernel type: {kernel_type}\n")
            f.write(f"  Mean accuracy: {res['accuracy_mean']:.4f}\n")
            f.write(f"  95% Confidence Interval: {res['accuracy_ci'][0]:.4f} - {res['accuracy_ci'][1]:.4f}\n")
            f.write(f"  Mean execution time: {res['time_mean']:.2f} seconds\n")
            
            f.write("\nDetailed accuracies:\n")
            for i, acc in enumerate(res['accuracies']):
                f.write(f"  Run {i+1}: {acc:.4f}\n")
        
        # Сравнение 
        f.write("\n" + "="*50 + "\n")
        f.write("COMPARISON WITH ORIGINAL PAPER\n")
        f.write("="*50 + "\n")
        f.write("Original ROCKET paper results for Ham dataset:\n")
        f.write("  Reported accuracy: 0.83\n")
        f.write("  Note: Results may vary due to different implementations\n")
        f.write("    and experimental setups.\n")
    
    print("\nExperiment completed successfully!")
    print(f"Results saved to directory: {args.output_dir}")

if __name__ == '__main__':
    main()
