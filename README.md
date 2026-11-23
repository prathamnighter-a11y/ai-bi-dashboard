# 📊 AI-Powered Business Intelligence Dashboard

An interactive Business Intelligence web app that lets you upload any CSV dataset and instantly get data exploration, smart visualizations, AI-generated insights, sales forecasting, and anomaly detection — all in one place.

Built with Python, Streamlit, Plotly, scikit-learn, and Google Gemini AI.

---

## 🚀 Live Demo

> Run locally — see setup instructions below.

---

## 🔍 What it does

Upload any CSV file and get a full 5-tab analytics dashboard:

| Tab | Feature | Description |
|---|---|---|
| 📋 Data | Explorer | View, search, and filter raw data. Summary statistics auto-generated. |
| 📊 Charts | Smart Visualizations | Histogram, Box Plot, Line Chart — auto-detects meaningful columns, ignores IDs |
| 🤖 AI Insights | Gemini AI Analysis | Sends your data to Google Gemini, returns 5 actionable business insights in plain English |
| 📈 Forecast | Sales Forecasting | Linear Regression model predicts future trend for any numeric metric |
| 🚨 Anomalies | Anomaly Detection | Isolation Forest ML flags unusual records (fraud, outliers, data errors) |

---

## 🧠 ML & AI Features

**Forecasting — Linear Regression**
- Trains on any numeric column (Sales, Revenue, Profit etc.)
- Predicts N future periods
- Shows trend direction with slope analysis

**Anomaly Detection — Isolation Forest**
- Unsupervised ML — no labels needed
- Adjustable sensitivity (contamination parameter)
- Flags and highlights anomalous records visually

**AI Insights — Google Gemini**
- Sends column names, sample rows, and summary statistics to Gemini API
- Returns 5 plain-English business insights
- Acts as a virtual senior business analyst

---

## 📁 Project Structure

```
ai-bi-dashboard/
│
├── app.py               # Main Streamlit application
├── train.csv            # Sample dataset — Superstore Sales (9,800 rows)
└── requirements.txt     # Python dependencies
```

---

## 🛠️ Tech Stack

- **Python** — core language
- **Streamlit** — web app framework
- **Plotly** — interactive charts
- **pandas / numpy** — data processing
- **scikit-learn** — Linear Regression + Isolation Forest
- **Google Gemini API** — AI-powered business insights

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/prathamnighter-a11y/ai-bi-dashboard.git
cd ai-bi-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

**5. Upload a CSV** from the sidebar and explore!

> To use AI Insights, you need a free Google Gemini API key from [aistudio.google.com](https://aistudio.google.com)

---

## 📊 Sample Dataset

Includes the **Superstore Sales dataset** (9,800 rows) with columns:
- Order details (ID, Date, Ship Mode)
- Customer info (Segment, Region, City, State)
- Product info (Category, Sub-Category, Product Name)
- **Sales** — the key metric for analysis

Try forecasting Sales trends and detecting anomalous orders right out of the box.

---

## ✨ Smart Column Detection

The dashboard automatically:
- Filters out ID columns, zip codes, phone numbers (not useful for analysis)
- Detects numeric columns with high unique ratios as sequential IDs (skipped)
- Only shows meaningful business metrics in charts, forecasting, and anomaly detection

---

## 👤 Author

Made by: Pratham Singh
College: Vivekananda Institute of Professional Studies, Delhi
Branch: B.Tech AI & Data Science
