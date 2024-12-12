import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型和标准化器
model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

svm_model = joblib.load('best_svm_model.pkl')
svm_scaler = joblib.load('svm_scaler.pkl')

# Streamlit 页面标题
st.title("医院数据分类预测")

st.write("请选择模型并输入特征值进行预测。")

# 用户选择模型（多选）
model_choices = st.sidebar.multiselect(
    '选择模型',
    ['随机森林分类器', 'SVM分类器'],
    default=['随机森林分类器', 'SVM分类器']  # 默认选择所有模型
)

# 用户输入部分
with st.form(key='input_form'):
    st.subheader("输入特征值")

    col1, col2 = st.columns(2)

    with col1:
        出院日龄 = st.number_input('出院日龄 (天)', min_value=-1, step=1, value=-1)
        有无子痫 = st.selectbox('有无子痫', [-1, 0, 1], format_func=lambda x: {-1: '未知', 0: '没有', 1: '有'}[x])
        n01AAAE2AA = st.selectbox('n01AAAE2AA', [-1, 0, 1], format_func=lambda x: {-1: '未知', 0: '没有', 1: '有'}[x])
        DIC = st.selectbox('DIC', [-1, 0, 1], format_func=lambda x: {-1: '未知', 0: '没有', 1: '有'}[x])
        开奶日龄 = st.number_input('开奶日龄 (天)', min_value=-1, step=1, value=-1)
        AE34 = st.selectbox('AE34', [-1, 0, 1], format_func=lambda x: {-1: '未知', 0: '没有', 1: '有'}[x])

    with col2:
        胆汁淤积症 = st.selectbox('胆汁淤积症', [-1, 0, 1], format_func=lambda x: {-1: '未知', 0: '没有', 1: '有'}[x])
        严重IVH = st.selectbox('严重IVH', [-1, 0, 1], format_func=lambda x: {-1: '未知', 0: '没有', 1: '有'}[x])
        听力 = st.selectbox('听力', [-1, 0, 1, 2, 3], format_func=lambda x: str(x))
        吸氧天数 = st.number_input('吸氧天数 (天)', min_value=-1, step=1, value=-1)

    # 提交按钮
    submit_button = st.form_submit_button(label='提交')

if submit_button:
    # 将用户输入转换为DataFrame
    user_input = {
        '出院日龄': [出院日龄],
        '有无子痫': [int(有无子痫)],
        'n01AAAE2AA': [int(n01AAAE2AA)],
        'DIC': [int(DIC)],
        '胆汁淤积症': [int(胆汁淤积症)],
        '开奶日龄': [开奶日龄],
        'AE34': [int(AE34)],
        '严重IVH': [int(严重IVH)],
        '听力': [int(听力)],
        '吸氧天数': [吸氧天数]
    }

    input_df = pd.DataFrame(user_input)

    predictions = {}

    if '随机森林分类器' in model_choices:
        input_scaled_rfc = scaler.transform(input_df)
        prediction_rfc = model.predict(input_scaled_rfc)
        predictions['随机森林分类器'] = prediction_rfc[0]

    if 'SVM分类器' in model_choices:
        input_scaled_svm = svm_scaler.transform(input_df)
        prediction_svm = svm_model.predict(input_scaled_svm)
        predictions['SVM分类器'] = prediction_svm[0]

    # 显示所有选定模型的预测结果
    st.subheader("预测结果")
    result_table = []

    for model_name, pred in predictions.items():
        result_table.append([model_name, pred])

    result_df = pd.DataFrame(result_table, columns=["模型", "预测转归2"])
    st.table(result_df)

# 添加一些样式
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">感谢您的使用！</p>', unsafe_allow_html=True)