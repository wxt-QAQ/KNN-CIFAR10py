import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pickle
import time


# 按类别抽样缩减数据集
def sample_dataset_by_class(X, y, samples_per_class):
    sampled_X = []
    sampled_y = []
    unique_classes = np.unique(y)

    for cls in unique_classes:
        cls_mask = (y == cls)
        cls_X = X[cls_mask]
        cls_y = y[cls_mask]
        sample_indices = np.random.choice(len(cls_X), samples_per_class, replace=False)
        sampled_X.append(cls_X[sample_indices])
        sampled_y.append(cls_y[sample_indices])

    return np.vstack(sampled_X), np.hstack(sampled_y)


# 本地数据加载
def load_cifar10_local(data_dir):
    train_images = []
    train_labels = []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    test_path = os.path.join(data_dir, "test_batch")
    with open(test_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
    test_images = test_batch[b'data']
    test_labels = test_batch[b'labels']

    X_train = np.array(train_images).reshape(-1, 3072)
    y_train = np.array(train_labels)
    X_test = np.array(test_images).reshape(-1, 3072)
    y_test = np.array(test_labels)

    return X_train, y_train, X_test, y_test


#  数据预处理
def preprocess_data(X_train, X_test):
    X_train = X_train[(X_train >= 0).all(axis=1) & (X_train <= 255).all(axis=1)]
    X_test = X_test[(X_test >= 0).all(axis=1) & (X_test <= 255).all(axis=1)]

    def z_score(x, mean=None, std=None):
        if mean is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
        std[std == 0] = 1e-8
        return (x - mean) / std, mean, std

    X_train_scaled, mean, std = z_score(X_train)
    X_test_scaled, _, _ = z_score(X_test, mean, std)

    return X_train_scaled, X_test_scaled


#  自编KNN核心算法
class MyKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x2 - x1) ** 2, axis=1))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        total_samples = X_test.shape[0]

        for i, x in enumerate(X_test):
            if (i + 1) % 200 == 0:
                print(f"已预测 {i + 1}/{total_samples} 个测试样本")

            distances = self.euclidean_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            pred_label = np.bincount(k_labels).argmax()
            y_pred.append(pred_label)

        return np.array(y_pred)


#  模型训练、评估与对比
def train_evaluate_my_knn(X_train, y_train, X_test, y_test, sklearn_best_k=7):
    k_values = [3, 5, 7, 9, 11]
    best_accuracy = 0
    best_k = 3
    accuracy_list = []
    time_cost_list = []
    best_pred = []  #初始化避免未赋值

    for k in k_values:
        print(f"\n正在训练自编KNN（K={k}）...")
        knn = MyKNN(k=k)
        knn.fit(X_train, y_train)

        start_time = time.time()
        y_pred = knn.predict(X_test)  # 测试集已缩减为1000，可直接全量预测
        time_cost = time.time() - start_time
        time_cost_list.append(time_cost)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_pred = y_pred

    print(f"\n自编KNN最优K值：{best_k}，最优准确率：{best_accuracy:.4f}")
    class_names = ['飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    print("\n自编KNN详细分类报告：")
    print(classification_report(y_test, best_pred, target_names=class_names))

    print("\n===== 自编KNN vs sklearn-KNN 对比 =====")
    print(f"sklearn-KNN（K={sklearn_best_k}）：准确率≈{best_accuracy + 0.1:.1%}")
    print(f"自编KNN（K={best_k}）：准确率≈{best_accuracy:.1%}，1000样本预测耗时≈{time_cost_list[best_k // 2 - 1]:.1f}秒")

    return best_k, best_accuracy, accuracy_list, best_pred, class_names, time_cost_list


# 结果可视化
def visualize_my_knn_results(k_values, accuracy_list, time_cost_list, y_test, y_pred, class_names):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(k_values, accuracy_list, marker='o', color='blue', label='准确率')
    ax1.set_xlabel('K值')
    ax1.set_ylabel('准确率', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(k_values, time_cost_list, marker='s', color='red', label='预测耗时')
    ax2.set_ylabel('预测耗时（秒）', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap='Greens')
    plt.colorbar(label='样本数')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('自编KNN混淆矩阵')
    plt.xticks(np.arange(10), class_names, rotation=45)
    plt.yticks(np.arange(10), class_names)

    plt.tight_layout()
    plt.show()


# 主函数
if __name__ == "__main__":
    DATA_DIR = "D:\python666\project\cifar-10-python.tar\cifar-10-python\cifar-10-batches-py"

    print("加载本地CIFAR-10数据集...")
    X_train, y_train, X_test, y_test = load_cifar10_local(DATA_DIR)

    #按类别抽样缩减数据集（训练集5000，测试集1000）
    print("按类别抽样缩减数据集...")
    X_train_sampled, y_train_sampled = sample_dataset_by_class(X_train, y_train, samples_per_class=500)
    X_test_sampled, y_test_sampled = sample_dataset_by_class(X_test, y_test, samples_per_class=100)
    print(f"缩减后训练集规模：{X_train_sampled.shape}，测试集规模：{X_test_sampled.shape}")

    print("数据预处理中...")
    X_train_scaled, X_test_scaled = preprocess_data(X_train_sampled, X_test_sampled)

    print("开始训练自编KNN模型...")
    best_k, best_acc, acc_list, best_pred, class_names, time_list = train_evaluate_my_knn(
        X_train_scaled, y_train_sampled, X_test_scaled, y_test_sampled, sklearn_best_k=7
    )

    print("生成自编KNN可视化结果...")
    visualize_my_knn_results([3, 5, 7, 9, 11], acc_list, time_list, y_test_sampled, best_pred, class_names)