import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

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

# 初始化随机森林分类器
rfc = RandomForestClassifier(random_state=42)

# 定义超参数网格以供搜索
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 使用RandomizedSearchCV进行超参数优化
random_search = RandomizedSearchCV(rfc, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# 在训练数据上执行随机搜索
random_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", random_search.best_params_)

# 使用最佳参数的模型在测试集上进行预测
best_rfc = random_search.best_estimator_
predictions = best_rfc.predict(X_test)

# 计算并打印准确率
accuracy = accuracy_score(y_test, predictions)
print(f"Test set accuracy: {accuracy:.4f}")

# 打印详细的分类报告
print(classification_report(y_test, predictions))

import joblib

# 保存最佳模型和标准化器到.pkl文件
model_filename = 'best_random_forest_model.pkl'
scaler_filename = 'scaler.pkl'

joblib.dump(best_rfc, model_filename)
print(f"Model saved to {model_filename}")

joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")