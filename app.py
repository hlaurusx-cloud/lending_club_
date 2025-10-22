import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="智能信用评级系统",
    page_icon="🏦",
    layout="wide"
)

# 标题和介绍
st.title("🏦 智能信用评级预测系统")
st.markdown("""
基于逻辑回归和神经网络的集成模型，用于预测个人/企业的信用等级。
""")

# 创建侧边栏
st.sidebar.header("🔧 系统配置")

# 模拟数据生成函数（实际使用时替换为真实数据）
def generate_sample_data(n_samples=20000):
    """生成模拟的信用数据"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'loan_amount': np.random.normal(20000, 10000, n_samples),
        'debt_to_income': np.random.normal(0.3, 0.15, n_samples),
        'employment_length': np.random.normal(5, 3, n_samples),
        'number_of_loans': np.random.randint(0, 10, n_samples),
        'delinquency_history': np.random.randint(0, 5, n_samples),
        'credit_history_length': np.random.normal(10, 5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 生成目标变量（信用等级 1-5）
    # 基于特征的线性组合加上一些噪声
    score = (
        0.1 * (df['income'] / 10000) +
        0.3 * (df['credit_score'] / 100) +
        -0.2 * df['debt_to_income'] * 10 +
        0.1 * (df['employment_length'] / 2) +
        -0.3 * df['delinquency_history'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # 将分数映射到信用等级 1-5
    df['credit_grade'] = pd.cut(score, bins=5, labels=[1, 2, 3, 4, 5])
    
    return df

# 数据加载和预处理
@st.cache_data
def load_and_preprocess_data():
    """加载和预处理数据"""
    # 生成模拟数据（实际使用时替换为你的数据）
    data = generate_sample_data(1000)  # 使用较小数据集用于演示
    
    # 分离特征和目标
    X = data.drop('credit_grade', axis=1)
    y = data['credit_grade']
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns, scaler

# 模型训练
@st.cache_resource
def train_models(X, y):
    """训练逻辑回归和神经网络模型"""
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练逻辑回归模型
    st.sidebar.info("训练逻辑回归模型中...")
    logreg_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    logreg_model.fit(X_train, y_train)
    
    # 训练神经网络模型
    st.sidebar.info("训练神经网络模型中...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    nn_model.fit(X_train, y_train)
    
    return logreg_model, nn_model, X_test, y_test

# 集成预测
def ensemble_predict(logreg_model, nn_model, X, method='average'):
    """集成模型预测"""
    if method == 'average':
        # 平均概率
        logreg_proba = logreg_model.predict_proba(X)
        nn_proba = nn_model.predict_proba(X)
        avg_proba = (logreg_proba + nn_proba) / 2
        return np.argmax(avg_proba, axis=1) + 1  # +1 因为信用等级从1开始
    else:
        # 投票机制
        logreg_pred = logreg_model.predict(X)
        nn_pred = nn_model.predict(X)
        # 简单投票（可以改进为加权投票）
        return np.round((logreg_pred.astype(int) + nn_pred.astype(int)) / 2).astype(int)

# 主应用
def main():
    # 加载数据
    X, y, feature_names, scaler = load_and_preprocess_data()
    
    # 训练模型
    logreg_model, nn_model, X_test, y_test = train_models(X, y)
    
    # 侧边栏 - 模型选择
    model_choice = st.sidebar.selectbox(
        "选择预测模型:",
        ["集成模型", "逻辑回归", "神经网络"]
    )
    
    # 侧边栏 - 特征重要性
    if st.sidebar.checkbox("显示特征重要性"):
        show_feature_importance(logreg_model, feature_names)
    
    # 主界面标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 单样本预测", 
        "📁 批量预测", 
        "📈 模型评估", 
        "ℹ️ 系统信息"
    ])
    
    with tab1:
        single_prediction_interface(logreg_model, nn_model, scaler, feature_names, model_choice)
    
    with tab2:
        batch_prediction_interface(logreg_model, nn_model, scaler, model_choice)
    
    with tab3:
        model_evaluation_interface(logreg_model, nn_model, X_test, y_test, model_choice)
    
    with tab4:
        system_info_interface()

def single_prediction_interface(logreg_model, nn_model, scaler, feature_names, model_choice):
    """单样本预测界面"""
    st.header("🔍 单样本信用评级预测")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("输入个人信息")
        
        # 创建输入表单
        with st.form("prediction_form"):
            age = st.slider("年龄", 18, 70, 35)
            income = st.number_input("年收入 ($)", 10000, 200000, 50000, step=1000)
            credit_score = st.slider("信用分数", 300, 850, 650)
            loan_amount = st.number_input("贷款金额 ($)", 1000, 100000, 20000, step=1000)
            debt_to_income = st.slider("债务收入比", 0.1, 1.0, 0.3, 0.05)
            employment_length = st.slider("工作年限", 0, 30, 5)
            number_of_loans = st.slider("现有贷款数量", 0, 10, 2)
            delinquency_history = st.slider("逾期记录次数", 0, 10, 0)
            credit_history_length = st.slider("信用历史长度 (年)", 0, 30, 10)
            
            submitted = st.form_submit_button("预测信用等级")
    
    with col2:
        st.subheader("预测结果")
        
        if submitted:
            # 准备输入数据
            input_data = np.array([[
                age, income, credit_score, loan_amount, debt_to_income,
                employment_length, number_of_loans, delinquency_history,
                credit_history_length
            ]])
            
            # 标准化
            input_scaled = scaler.transform(input_data)
            
            # 根据选择的模型进行预测
            if model_choice == "逻辑回归":
                prediction = logreg_model.predict(input_scaled)[0]
                probability = logreg_model.predict_proba(input_scaled)[0]
            elif model_choice == "神经网络":
                prediction = nn_model.predict(input_scaled)[0]
                probability = nn_model.predict_proba(input_scaled)[0]
            else:  # 集成模型
                prediction = ensemble_predict(logreg_model, nn_model, input_scaled)[0]
                # 对于集成模型，我们计算平均概率
                logreg_proba = logreg_model.predict_proba(input_scaled)[0]
                nn_proba = nn_model.predict_proba(input_scaled)[0]
                probability = (logreg_proba + nn_proba) / 2
            
            # 显示结果
            display_prediction_result(prediction, probability)
            
            # 显示特征影响分析
            st.subheader("📋 特征影响分析")
            feature_analysis = analyze_feature_impact(
                logreg_model, input_scaled[0], feature_names
            )
            st.dataframe(feature_analysis)

def display_prediction_result(prediction, probability):
    """显示预测结果"""
    # 信用等级描述
    grade_descriptions = {
        1: "优秀 - 风险极低",
        2: "良好 - 风险较低", 
        3: "一般 - 中等风险",
        4: "关注 - 风险较高",
        5: "不良 - 高风险"
    }
    
    # 颜色映射
    grade_colors = {
        1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "⚫"
    }
    
    # 显示主要结果
    st.metric(
        label="预测信用等级",
        value=f"{grade_colors[prediction]} 等级 {prediction}",
        delta=grade_descriptions[prediction]
    )
    
    # 显示概率分布
    st.subheader("各等级预测概率")
    
    prob_df = pd.DataFrame({
        '信用等级': [f'等级 {i}' for i in range(1, 6)],
        '概率': probability,
        '描述': [grade_descriptions[i] for i in range(1, 6)]
    })
    
    # 显示概率条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    bars = ax.bar(prob_df['信用等级'], prob_df['概率'] * 100, color=colors)
    
    # 在条形上添加数值
    for bar, prob in zip(bars, prob_df['概率']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{prob*100:.1f}%', ha='center', va='bottom')
    
    ax.set_ylabel('概率 (%)')
    ax.set_ylim(0, 100)
    ax.set_title('各信用等级预测概率分布')
    st.pyplot(fig)

def analyze_feature_impact(model, input_features, feature_names):
    """分析特征对预测的影响"""
    if hasattr(model, 'coef_'):
        # 对于逻辑回归模型
        coefficients = model.coef_[0]  # 取第一个类的系数（多分类时可能需要调整）
        impact = coefficients * input_features
        
        impact_df = pd.DataFrame({
            '特征': feature_names,
            '原始值': input_features,
            '系数权重': coefficients,
            '影响程度': impact,
            '影响方向': ['正向' if x > 0 else '负向' for x in impact]
        })
        
        return impact_df.sort_values('影响程度', key=abs, ascending=False)
    else:
        return pd.DataFrame({'消息': ['特征影响分析仅适用于逻辑回归模型']})

def batch_prediction_interface(logreg_model, nn_model, scaler, model_choice):
    """批量预测界面"""
    st.header("📁 批量信用评级预测")
    
    uploaded_file = st.file_uploader(
        "上传CSV文件进行批量预测", 
        type=['csv'],
        help="请确保文件包含所需的特征列"
    )
    
    if uploaded_file is not None:
        try:
            # 读取数据
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"成功读取数据，共 {len(batch_data)} 条记录")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(batch_data.head())
            
            if st.button("开始批量预测"):
                # 这里应该添加数据验证和预处理
                # 假设数据已经包含所需特征
                
                # 标准化数据
                batch_scaled = scaler.transform(batch_data)
                
                # 批量预测
                if model_choice == "逻辑回归":
                    predictions = logreg_model.predict(batch_scaled)
                elif model_choice == "神经网络":
                    predictions = nn_model.predict(batch_scaled)
                else:  # 集成模型
                    predictions = ensemble_predict(logreg_model, nn_model, batch_scaled)
                
                # 添加预测结果到数据
                batch_data['预测信用等级'] = predictions
                
                # 显示结果
                st.subheader("批量预测结果")
                st.dataframe(batch_data)
                
                # 统计结果
                st.subheader("预测结果统计")
                result_counts = batch_data['预测信用等级'].value_counts().sort_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(result_counts)
                
                with col2:
                    fig, ax = plt.subplots()
                    result_counts.plot(kind='bar', ax=ax, color=['green', 'lightgreen', 'yellow', 'orange', 'red'])
                    ax.set_title('信用等级分布')
                    ax.set_xlabel('信用等级')
                    ax.set_ylabel('数量')
                    st.pyplot(fig)
                
                # 下载结果
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="下载预测结果",
                    data=csv,
                    file_name="信用评级预测结果.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"处理文件时出错: {str(e)}")

def model_evaluation_interface(logreg_model, nn_model, X_test, y_test, model_choice):
    """模型评估界面"""
    st.header("📈 模型性能评估")
    
    # 根据选择的模型进行预测
    if model_choice == "逻辑回归":
        y_pred = logreg_model.predict(X_test)
        model_name = "逻辑回归"
    elif model_choice == "神经网络":
        y_pred = nn_model.predict(X_test)
        model_name = "神经网络"
    else:  # 集成模型
        y_pred = ensemble_predict(logreg_model, nn_model, X_test)
        model_name = "集成模型"
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("准确率", f"{accuracy:.3f}")
    
    with col2:
        st.metric("测试样本数", len(y_test))
    
    with col3:
        st.metric("模型", model_name)
    
    # 混淆矩阵
    st.subheader("混淆矩阵")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(f'{model_name} - 混淆矩阵')
    st.pyplot(fig)
    
    # 分类报告
    st.subheader("详细分类报告")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

def show_feature_importance(model, feature_names):
    """显示特征重要性"""
    st.sidebar.subheader("特征重要性")
    
    if hasattr(model, 'coef_'):
        # 逻辑回归的特征重要性
        importance = np.abs(model.coef_[0])
        feature_imp = pd.DataFrame({
            '特征': feature_names,
            '重要性': importance
        }).sort_values('重要性', ascending=False)
        
        for _, row in feature_imp.head().iterrows():
            st.sidebar.write(f"• {row['特征']}: {row['重要性']:.3f}")
    else:
        st.sidebar.info("神经网络模型的特征重要性分析较复杂")

def system_info_interface():
    """系统信息界面"""
    st.header("ℹ️ 系统信息")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("系统架构")
        st.markdown("""
        - **前端**: Streamlit
        - **机器学习框架**: Scikit-learn
        - **主要算法**: 
          - 逻辑回归 (多分类)
          - 神经网络 (MLP)
          - 集成学习
        - **数据处理**: Pandas, NumPy
        """)
    
    with col2:
        st.subheader("模型特性")
        st.markdown("""
        - ✅ 支持单样本实时预测
        - ✅ 支持批量数据处理
        - ✅ 模型性能可视化
        - ✅ 特征影响分析
        - ✅ 结果导出功能
        """)
    
    st.subheader("使用说明")
    st.markdown("""
    1. **单样本预测**: 在"单样本预测"标签页中输入个人信用信息
    2. **批量预测**: 在"批量预测"标签页中上传CSV文件
    3. **模型评估**: 查看不同模型的性能指标
    4. **结果解读**: 信用等级1-5，数字越小信用越好
    """)

# 运行应用
if __name__ == "__main__":
    main()
