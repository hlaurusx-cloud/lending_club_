import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm 
import re 

# --- 1. 模型及特征加载函数 ---
@st.cache_resource
def load_assets():
    """加载模型和特征列表，只在应用启动时执行一次"""
    try:
        # 加载最终模型
        model = joblib.load('lending_club_model.pkl')
        # 加载特征列表
        model_features = joblib.load('model_features.pkl')
        
        # ⚠️ 如果您也保存了 StandardScaler 或其他预处理器，请在此处加载
        
        return model, model_features
    except FileNotFoundError:
        st.error("🚨 缺少模型或特征文件。请确保 'lending_club_model.pkl' 和 'model_features.pkl' 文件已准备好。")
        st.stop()
    except Exception as e:
        st.error(f"🚨 资源加载失败。请检查文件是否损坏或格式是否正确: {e}")
        st.stop()


# --- 2. 输入数据预处理函数 ---
def preprocess_data(input_df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    将用户输入数据转换为模型可以接受的格式。
    必须精确复现训练时的所有预处理步骤！
    """
    
    # 1. 数值特征预处理 
    if 'int_rate' in input_df.columns:
        input_df['int_rate'] = input_df['int_rate'].str.replace('%', '').astype(float) / 100
        
    if 'term' in input_df.columns:
        input_df['term'] = input_df['term'].str.replace(' months', '').astype(int)
        
    if 'emp_length' in input_df.columns:
        emp_mapping = {
            '< 1 year': 0.5, '10+ years': 10.0, '9 years': 9.0, '8 years': 8.0, 
            '7 years': 7.0, '6 years': 6.0, '5 years': 5.0, '4 years': 4.0, 
            '3 years': 3.0, '2 years': 2.0, '1 year': 1.0, 'n/a': np.nan 
        }
        input_df['emp_length'] = input_df['emp_length'].map(emp_mapping).fillna(0) 
        
    # 2. 独热编码 (One-Hot Encoding)
    categorical_cols = ['grade', 'home_ownership', 'purpose'] # 请根据您的模型特征确认
    
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
    
    # 3. 匹配模型特征列表 (关键步骤)
    final_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    # 复制已存在的列值
    for col in final_data.columns:
        if col in input_df.columns:
            final_data[col] = input_df[col].iloc[0]
            
    # 4. 检查并添加截距项 (如果模型需要的话)
    if 'Intercept' in model_features and 'Intercept' not in final_data.columns:
        # Statsmodels 默认添加截距，如果您的特征列表中有 'Intercept'，则需要手动添加
        final_data['Intercept'] = 1.0
        
    # 5. 确保列顺序和名称完全一致
    final_data = final_data[model_features]
            
    return final_data


# --- 3. Streamlit UI 及主函数 ---
def main():
    st.set_page_config(page_title="Lending Club 违约预测", layout="sidebar")
    st.title('💰 Lending Club 贷款违约风险预测系统')
    st.markdown('---')
    
    # 加载模型和特征
    model, model_features = load_assets()

    st.sidebar.header('贷款申请信息输入')
    
    # --- 用户输入字段 (请根据您的模型特征进行调整) ---
    loan_amnt = st.sidebar.slider('贷款金额 (loan_amnt)', 1000, 40000, 10000)
    int_rate_str = st.sidebar.text_input('贷款利率 (%) (int_rate)', '7.97%')
    term_str = st.sidebar.selectbox('还款期限 (term)', ('36 months', '60 months'))
    grade = st.sidebar.selectbox('贷款等级 (grade)', ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
    annual_inc = st.sidebar.number_input('年收入 (annual_inc)', 10000.0, 500000.0, 50000.0, step=1000.0)
    dti = st.sidebar.number_input('债务收入比 (dti)', 0.0, 50.0, 15.0, step=0.1)
    home_ownership = st.sidebar.selectbox('住房情况 (home_ownership)', ('MORTGAGE', 'RENT', 'OWN', 'OTHER'))
    purpose = st.sidebar.selectbox('贷款目的 (purpose)', ('debt_consolidation', 'credit_card', 'home_improvement', 'other'))
    installment = st.sidebar.number_input('月供金额 (installment)', 50.0, 1500.0, 300.0, step=10.0)
    emp_length_str = st.sidebar.selectbox('雇佣年限 (emp_length)', ('10+ years', '2 years', '5 years', '1 year', '< 1 year', 'n/a'))
    
    # --- 预测按钮 ---
    if st.button('执行违约风险预测'):
        # 1. 组合用户输入为 DataFrame
        input_data = {
            'loan_amnt': [loan_amnt], 'int_rate': [int_rate_str], 'term': [term_str],
            'grade': [grade], 'annual_inc': [annual_inc], 'dti': [dti],
            'home_ownership': [home_ownership], 'purpose': [purpose],
            'installment': [installment], 'emp_length': [emp_length_str]
        }
        input_df = pd.DataFrame(input_data)
        
        # 2. 预处理
        with st.spinner('正在预处理数据并进行预测...'):
            processed_data = preprocess_data(input_df, model_features)
        
        # 3. 预测 실행
        try:
            # Statsmodels GLM 模型使用 .predict() 方法返回概率 (违约概率)
            prediction_prob = model.predict(processed_data)[0]
            default_prob_percent = round(prediction_prob * 100, 2)
            
            st.markdown('## 📊 预测结果')
            
            threshold = 0.35 # 违约阈值
            
            if prediction_prob >= threshold: 
                st.error(f'### ❌ 预测违约风险：高风险')
                st.warning(f'**预计违约概率:** **{default_prob_percent}%**')
                st.markdown("建议拒绝或进一步审查此贷款申请，风险较高。")
            else:
                st.success(f'### ✅ 预测违约风险：低风险')
                st.info(f'**预计违约概率:** **{default_prob_percent}%**')
                st.markdown("贷款申请通过的可能性较高，风险较低。")
                
            st.markdown('---')
            st.subheader("模型输入数据 (已预处理)")
            st.dataframe(processed_data.T, use_container_width=True) 

        except Exception as e:
            st.error(f"⚠️ 预测过程中发生错误。请检查预处理逻辑是否与训练时一致: {e}")

if __name__ == '__main__':
    main()
