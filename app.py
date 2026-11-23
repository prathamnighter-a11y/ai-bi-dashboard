import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .block-container { padding: 2rem 3rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 1rem;
    }
    .metric-card h3 { color: #a0aec0; font-size: 0.85rem; margin: 0; }
    .metric-card h2 { color: #ffffff; font-size: 1.8rem; margin: 0.3rem 0 0 0; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #4f8ef7;
        border-bottom: 2px solid #4f8ef7;
        padding-bottom: 0.4rem;
        margin: 2rem 0 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f8ef7, #7c6ef7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    div[data-testid="stSidebar"] { background: #1a1d2e; }
    </style>
""", unsafe_allow_html=True)

# --- Smart Column Classifier ---
def classify_columns(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()

    id_keywords = ['id', 'code', 'zip', 'postal', 'phone', 'fax', 'number', 'no', 'num', 'index']

    useful_numeric = []
    for col in numeric_cols:
        col_lower = col.lower()
        unique_ratio = df[col].nunique() / len(df)
        is_id_name = any(kw in col_lower for kw in id_keywords)
        is_sequential = unique_ratio > 0.85
        if not is_id_name and not is_sequential:
            useful_numeric.append(col)

    return useful_numeric, categorical_cols

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("BI Dashboard")
    st.markdown("---")
    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🤖 AI Settings")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("Upload any CSV file to get instant analysis, AI insights, forecasting and anomaly detection.")

# --- Welcome Screen ---
if uploaded_file is None:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 3rem;'>
            <h1 style='color:#4f8ef7; font-size:3rem;'>📊 BI Dashboard</h1>
            <p style='color:#a0aec0; font-size:1.2rem;'>Your AI-powered Business Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("📋", "Data Explorer", "View and filter your raw data instantly"),
        ("📊", "Smart Charts", "Auto-generated visualizations"),
        ("🤖", "AI Insights", "Plain-English business insights"),
        ("🚨", "Anomaly Detection", "Spot unusual patterns automatically"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <div style='font-size:2rem;'>{icon}</div>
                <h3 style='color:#ffffff; font-size:1rem;'>{title}</h3>
                <p style='color:#a0aec0; font-size:0.8rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><p style='text-align:center; color:#a0aec0;'>👈 Upload a CSV file from the sidebar to get started</p>", unsafe_allow_html=True)

else:
    df = pd.read_csv(uploaded_file)
    useful_numeric_cols, categorical_cols = classify_columns(df)

    # --- Top Metrics ---
    st.markdown("<div class='section-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Total Rows", f"{df.shape[0]:,}"),
        ("Total Columns", f"{df.shape[1]}"),
        ("Analysis Columns", f"{len(useful_numeric_cols)}"),
        ("Missing Values", f"{df.isnull().sum().sum():,}"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{label}</h3>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)

    if not useful_numeric_cols:
        st.error("⚠️ No meaningful numeric columns found for analysis. Make sure your CSV has columns like Sales, Profit, Revenue etc.")
        st.stop()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data", "📊 Charts", "🤖 AI Insights", "📈 Forecast", "🚨 Anomalies"])

    # --- Tab 1: Raw Data ---
    with tab1:
        st.markdown("<div class='section-header'>📋 Raw Data</div>", unsafe_allow_html=True)
        search = st.text_input("🔍 Search / Filter", placeholder="Type to filter rows...")
        if search:
            filtered = df[df.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)]
            st.dataframe(filtered, use_container_width=True)
            st.caption(f"Showing {len(filtered)} of {len(df)} rows")
        else:
            st.dataframe(df, use_container_width=True)
        st.markdown("<div class='section-header'>📈 Summary Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df[useful_numeric_cols].describe(), use_container_width=True)

    # --- Tab 2: Charts ---
    with tab2:
        st.markdown("<div class='section-header'>📊 Visualizations</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            selected_num = st.selectbox("Select metric to visualize", useful_numeric_cols)
        with col2:
            chart_type = st.selectbox("Chart type", ["Histogram", "Box Plot", "Line Chart"])

        if chart_type == "Histogram":
            fig = px.histogram(df, x=selected_num, template="plotly_dark",
                               color_discrete_sequence=["#4f8ef7"],
                               title=f"Distribution of {selected_num}")
        elif chart_type == "Box Plot":
            fig = px.box(df, y=selected_num, template="plotly_dark",
                         color_discrete_sequence=["#4f8ef7"],
                         title=f"Box Plot of {selected_num}")
        elif chart_type == "Line Chart":
            fig = px.line(df, y=selected_num, template="plotly_dark",
                          color_discrete_sequence=["#4f8ef7"],
                          title=f"Trend of {selected_num}")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        if categorical_cols:
            st.markdown("---")
            st.markdown("#### 📊 Category Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                cat = st.selectbox("Category column", categorical_cols)
            with col2:
                num = st.selectbox("Metric", useful_numeric_cols, key="cat_num")
            fig2 = px.bar(df, x=cat, y=num, template="plotly_dark",
                          color_discrete_sequence=["#7c6ef7"],
                          title=f"{num} by {cat}")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 3: AI Insights ---
    with tab3:
        st.markdown("<div class='section-header'>🤖 AI Business Insights</div>", unsafe_allow_html=True)
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API Key in the sidebar to use this feature.")
        else:
            if st.button("✨ Generate AI Insights"):
                summary = df[useful_numeric_cols].describe().to_string()
                columns = ", ".join(df.columns.tolist())
                sample = df.head(3).to_string()
                prompt = f"""
                You are a senior business analyst. Analyze this dataset and give 5 clear,
                actionable business insights in plain English.
                Columns: {columns}
                Sample data: {sample}
                Statistical summary: {summary}
                Give insights as numbered points. Be specific and practical.
                """
                try:
                    with st.spinner("🤖 Analysing your data..."):
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=0.7)
                        )
                        st.success("✅ Insights generated!")
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # --- Tab 4: Forecasting ---
    with tab4:
        st.markdown("<div class='section-header'>📈 Sales Forecasting</div>", unsafe_allow_html=True)
        st.caption("ℹ️ Only meaningful business metrics are available here (IDs and codes are excluded automatically)")
        col1, col2 = st.columns(2)
        with col1:
            forecast_col = st.selectbox("Metric to forecast", useful_numeric_cols, key="forecast")
        with col2:
            periods = st.slider("Future periods to predict", 5, 100, 20)

        if st.button("📈 Generate Forecast"):
            data = df[forecast_col].dropna().reset_index(drop=True)
            X = np.array(range(len(data))).reshape(-1, 1)
            y = data.values
            model_lr = LinearRegression()
            model_lr.fit(X, y)
            future_X = np.array(range(len(data), len(data) + periods)).reshape(-1, 1)
            future_y = model_lr.predict(future_X)
            actual = pd.DataFrame({"Index": range(len(data)), "Value": y, "Type": "Actual"})
            predicted = pd.DataFrame({"Index": range(len(data), len(data) + periods), "Value": future_y, "Type": "Forecast"})
            combined = pd.concat([actual, predicted])
            fig3 = px.line(combined, x="Index", y="Value", color="Type",
                           title=f"Forecast for {forecast_col}",
                           template="plotly_dark",
                           color_discrete_map={"Actual": "#4f8ef7", "Forecast": "#f76f4f"})
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
            trend = "growth 📈" if model_lr.coef_[0] > 0 else "decline 📉"
            st.info(f"📊 Trend: The model predicts **{trend}** with a slope of **{model_lr.coef_[0]:.2f}** per period.")

    # --- Tab 5: Anomaly Detection ---
    with tab5:
        st.markdown("<div class='section-header'>🚨 Anomaly Detection</div>", unsafe_allow_html=True)
        st.caption("ℹ️ Only meaningful business metrics are available here (IDs and codes are excluded automatically)")
        col1, col2 = st.columns(2)
        with col1:
            anomaly_col = st.selectbox("Metric to analyse", useful_numeric_cols, key="anomaly")
        with col2:
            contamination = st.slider("Sensitivity", 0.01, 0.2, 0.05,
                                      help="Higher = more anomalies flagged")

        if st.button("🚨 Detect Anomalies"):
            data_anomaly = df[[anomaly_col]].dropna().copy()
            iso = IsolationForest(contamination=contamination, random_state=42)
            data_anomaly["Anomaly"] = iso.fit_predict(data_anomaly[[anomaly_col]])
            data_anomaly["Status"] = data_anomaly["Anomaly"].map({1: "Normal", -1: "Anomaly"})
            data_anomaly["Index"] = range(len(data_anomaly))
            fig4 = px.scatter(data_anomaly, x="Index", y=anomaly_col, color="Status",
                              title=f"Anomaly Detection on {anomaly_col}",
                              template="plotly_dark",
                              color_discrete_map={"Normal": "#4f8ef7", "Anomaly": "#f76f4f"})
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
            anomaly_count = (data_anomaly["Anomaly"] == -1).sum()
            col1, col2 = st.columns(2)
            with col1:
                st.warning(f"⚠️ **{anomaly_count}** anomalies detected out of **{len(data_anomaly)}** records")
            with col2:
                st.info(f"✅ **{len(data_anomaly) - anomaly_count}** normal records")
            st.markdown("#### 🔍 Anomalous Records")
            st.dataframe(data_anomaly[data_anomaly["Anomaly"] == -1][[anomaly_col, "Status"]], use_container_width=True)