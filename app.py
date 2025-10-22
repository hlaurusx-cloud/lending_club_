import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix, accuracy_score)
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="信用评分 Demo",
                   page_icon="💳",
                   layout="centered",
                   initial_sidebar_state="collapsed")

# ---------- 全局样式 ----------
st.markdown("""
<style>
    .stButton>button {
        font-size: 1.2rem;
        padding: .5rem 2rem;
        border-radius: 12px;
        background: #007AFF;
        color: white;
        border: none;
    }
    h1, h2, h3 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- 标题 ----------
st.title("💳 智能信用评分")
st.markdown("上传一份带标签的贷款数据，3 秒完成训练与报告。")

# ---------- 上传 ----------
uploaded = st.file_uploader("选择 CSV", type="csv")
if uploaded is None:
    st.stop()

# ---------- 自动训练 ----------
@st.cache_data(show_spinner=False)
def train_pipeline(df):
    # 极简清洗：自动识别数字列 & 缺失值填充
    num_cols = df.select_dtypes(include='number').columns
    X = df[num_cols].drop(columns=['y'], errors='ignore')
    y = df['y']  # 标签列必须叫 y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(
        scaler.transform(X_train), y_train)
    pred = model.predict(scaler.transform(X_test))
    return model, scaler, X_test, y_test, pred, X.columns

with st.spinner("训练ing..."):
    df = pd.read_csv(uploaded)
    model, scaler, X_test, y_test, pred, features = train_pipeline(df)

# ---------- 结果 ----------
st.success("训练完成！")
col1, col2, col3 = st.columns(3)
acc = accuracy_score(y_test, pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, pred, average='binary')
col1.metric("准确率", f"{acc:.1%}")
col2.metric("精确率", f"{precision:.1%}")
col3.metric("召回率", f"{recall:.1%}")

# 混淆矩阵
fig, ax = plt.subplots(figsize=(3, 3))
sns.heatmap(confusion_matrix(y_test, pred),
            annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("预测"), ax.set_ylabel("真实")
st.pyplot(fig)

# ---------- 交互预测 ----------
st.header("🔮 单笔预测")
with st.form("predict"):
    inputs = {f: st.number_input(f, value=0.0) for f in features}
    if st.form_submit_button("预测"):
        sample = pd.DataFrame([inputs])
        proba = model.predict_proba(scaler.transform(sample))[0, 1]
        st.write(f"违约概率：**{proba:.1%}**")

# ---------- 下载报告 ----------
report = io.StringIO()
report.write("信用评分报告\n")
report.write(f"准确率: {acc:.1%}\n")
report.write(f"精确率: {precision:.1%}\n")
report.write(f"召回率: {recall:.1%}\n")
report.write("混淆矩阵:\n")
report.write(str(confusion_matrix(y_test, pred)))
b64 = base64.b64encode(report.getvalue().encode()).decode()
href = f'<a href="data:text/plain;base64,{b64}" download="report.txt">📥 下载报告</a>'
st.markdown(href, unsafe_allow_html=True)
