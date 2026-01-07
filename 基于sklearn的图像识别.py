import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import pickle

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


#  本地数据加载
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# 模型训练与评估
def train_evaluate_sklearn_knn(X_train, y_train, X_test, y_test):
    k_values = [3, 5, 7, 9, 11]
    best_accuracy = 0
    best_k = 3
    accuracy_list = []
    best_pred = []  #初始化避免未赋值

    for k in k_values:
        print(f"正在训练K={k}的KNN模型...")
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_pred = y_pred

    print(f"\n最优K值：{best_k}，最优准确率：{best_accuracy:.4f}")
    class_names = ['飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    print("\n详细分类报告：")
    print(classification_report(y_test, best_pred, target_names=class_names))

    return best_k, best_accuracy, accuracy_list, best_pred, class_names


#  结果可视化
def visualize_results(k_values, accuracy_list, y_test, y_pred, class_names):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, accuracy_list, marker='o', linewidth=2)
    plt.xlabel('K值')
    plt.ylabel('准确率')
    plt.title('不同K值下KNN模型准确率')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar(label='样本数')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
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

    print("开始训练sklearn-KNN模型...")
    best_k, best_acc, acc_list, best_pred, class_names = train_evaluate_sklearn_knn(
        X_train_scaled, y_train_sampled, X_test_scaled, y_test_sampled
    )

    print("生成可视化结果...")
    visualize_results([3, 5, 7, 9, 11], acc_list, y_test_sampled, best_pred, class_names)

