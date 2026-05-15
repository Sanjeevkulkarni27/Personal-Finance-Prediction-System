# 💹 Personal Finance Prediction System

A machine learning web application that predicts whether a financial transaction is **Income** or **Expense** using a trained Random Forest classifier.

---

## 🚀 Features

- **ML-Powered Prediction** — Random Forest model trained on real transaction data
- **Confidence Score** — Shows model probability for each prediction
- **Prediction History** — Tracks last 10 predictions in-session
- **Premium Dark UI** — Built with Streamlit and custom CSS
- **Instant Startup** — Model is cached, no retraining on reload

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| ML | Scikit-learn (Random Forest) |
| Data | Pandas, NumPy |
| UI | Streamlit |
| Styling | Custom CSS (glassmorphism dark theme) |

---

## 📂 Project Structure

```
Personal Finance Prediction System/
│
├── streamlit_app.py                  # Main Streamlit app (run this)
├── personal_finance_prediction_system.py  # Original ML pipeline script
├── Personal_Finance_Dataset.csv      # Transaction dataset
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Sanjeevkulkarni27/personal-finance-prediction-system.git
cd personal-finance-prediction-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python -m streamlit run streamlit_app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🧠 How It Works

1. **Data Preprocessing** — Cleans transactions, extracts Month & Day from dates, applies log-transform on Amount
2. **Feature Engineering** — Features: `log(Amount)`, `Month`, `Day`
3. **Model Training** — Random Forest with balanced class weights (~94% accuracy)
4. **Prediction** — Enter an amount + date → get Income/Expense prediction with confidence %

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| Random Forest ✅ | ~94% |
| Naive Bayes | ~90% |
| Logistic Regression | ~72% |
| KNN | ~86% |
| SVM | ~83% |

---

## 📸 Screenshot

> Dark themed, 3-panel layout: Transaction Inputs · Prediction Result · History

---

## 📄 License

MIT License — free to use and modify.
