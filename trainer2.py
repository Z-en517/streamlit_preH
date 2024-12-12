import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import reciprocal, uniform
import joblib

# 加载数据
file_path = '医院数据集新1210.xlsx'
data = pd.read_excel(file_path)

# 指定要使用的十个特征和目标变量
selected_predictor_names = ['出院日龄', '有无子痫', 'n01AAAE2AA', 'DIC', '胆汁淤积症',
                            '开奶日龄', 'AE34', '严重IVH', '听力', '吸氧天数']
target = '转归2'

# 提取特征和目标变量
X = data[selected_predictor_names]
y = data[target]

# 数据标准化（如果需要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化SVM分类器
svm_clf = SVC(probability=True, random_state=42)

# 定义超参数网格以供搜索
param_dist = {
    'kernel': ['linear', 'rbf', 'poly'],  # 尝试不同的核函数
    'C': reciprocal(20, 200000),          # C参数的范围
    'gamma': ['scale', 'auto'] + list(uniform(loc=0, scale=1).rvs(10)),  # gamma参数的范围
    'degree': [2, 3, 4],                  # 如果使用多项式核，则尝试不同的度数
}

# 使用RandomizedSearchCV进行超参数优化
random_search = RandomizedSearchCV(
    estimator=svm_clf,
    param_distributions=param_dist,
    n_iter=50,  # 可以根据计算资源调整此参数
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# 在训练数据上执行随机搜索
random_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)

# 使用最佳参数的模型在测试集上进行预测
best_svm_clf = random_search.best_estimator_
predictions = best_svm_clf.predict(X_test)

# 计算并打印准确率
accuracy = accuracy_score(y_test, predictions)
print(f"Test set accuracy: {accuracy:.4f}")

# 打印详细的分类报告
print(classification_report(y_test, predictions))

# 保存SVM模型和标准化器
svm_model_filename = 'best_svm_model.pkl'
svm_scaler_filename = 'svm_scaler.pkl'

joblib.dump(best_svm_clf, svm_model_filename)
print(f"SVM Model saved to {svm_model_filename}")

joblib.dump(scaler, svm_scaler_filename)  # 假设SVM使用的标准化器与随机森林相同
print(f"SVM Scaler saved to {svm_scaler_filename}")