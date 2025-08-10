import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_UCR_UEA_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class ROCKET:
    def __init__(self, num_kernels=10000, kernel_lengths=[7, 9, 11], kernel_type='normal', random_state=None):
        self.num_kernels = num_kernels
        self.kernel_lengths = kernel_lengths
        self.kernel_type = kernel_type
        self.kernels = []
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y=None):
        n_series, n_timesteps = X.shape
        self.kernels = []
        
        max_length = max(self.kernel_lengths)
        max_dilation = max(1, int(np.log2((n_timesteps - 1) / (max_length - 1))) if max_length > 1 else 0
        
        for _ in range(self.num_kernels):
            L = np.random.choice(self.kernel_lengths)
            
            # Generate kernel
            if self.kernel_type == 'normal':
                kernel = np.random.normal(0, 1, L)
                kernel = (kernel - np.mean(kernel)) / (np.std(kernel) + 1e-8)
            elif self.kernel_type == 'binary':
                kernel = np.random.choice([-1, 1], size=L)
            elif self.kernel_type == 'ternary':
                kernel = np.random.choice([-1, 0, 1], size=L, p=[1/3, 1/3, 1/3])
            
            # Generate dilation and bias
            dilation = 2 ** np.random.uniform(0, max_dilation) if max_dilation > 0 else 1
            dilation = int(np.round(dilation))
            bias = np.random.uniform(-1, 1)
            
            self.kernels.append((kernel, bias, dilation))
        
        return self
    
    def transform(self, X):
        n_series = X.shape[0]
        features = np.zeros((n_series, 2 * self.num_kernels))
        
        for i, (kernel, bias, dilation) in enumerate(tqdm(self.kernels, desc="Applying kernels")):
            for j in range(n_series):
                conv = self.apply_kernel(X[j], kernel, bias, dilation)
                if len(conv) > 0:
                    features[j, 2*i] = np.max(conv)
                    features[j, 2*i+1] = np.mean(conv > 0)
        
        return features
    
    def apply_kernel(self, x, kernel, bias, dilation):
        L = len(kernel)
        n_timesteps = len(x)
        num_windows = n_timesteps - (L - 1) * dilation
        
        if num_windows < 1:
            return np.array([])
        
        conv = np.zeros(num_windows)
        for t in range(num_windows):
            start = t * dilation
            end = start + L * dilation
            segment = x[start:end:dilation]
            conv[t] = np.dot(segment, kernel) + bias
        
        return conv
    pass

def load_data(dataset_name='Ham'):
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split='train', return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split='test', return_X_y=True)
    
    # Convert to 2D array
    X_train = np.vstack(X_train.iloc[:, 0].apply(lambda x: x.values))
    X_test = np.vstack(X_test.iloc[:, 0].apply(lambda x: x.values))
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T
    
    return X_train, y_train, X_test, y_test
    pass

def run_experiment(kernel_type, n_runs=10, num_kernels=10000):
    accuracies = []
    
    for run in range(n_runs):
        rocket = ROCKET(num_kernels=num_kernels, kernel_type=kernel_type, random_state=run)
        rocket.fit(X_train)
        
        X_train_features = rocket.transform(X_train)
        X_test_features = rocket.transform(X_test)
        
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 100), normalize=True)
        model.fit(X_train_features, y_train)
        
        y_pred = model.predict(X_test_features)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Run {run+1}/{n_runs} - Accuracy: {acc:.4f}")
    
    return accuracies
    pass

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    
    results = {}
    kernel_types = ['normal', 'binary', 'ternary']
    
    for k_type in kernel_types:
        print(f"\nRunning experiments for {k_type} kernels...")
        accs = run_experiment(k_type)
        results[k_type] = accs
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for k_type, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        plt.errorbar(k_type, mean_acc, yerr=1.96*std_acc/np.sqrt(len(accs)), fmt='o', capsize=10)
    
    plt.title("ROCKET Performance on Ham Dataset (95% CI)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("rocket_results.png")
    plt.show()
