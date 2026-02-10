# 🚀 Quick Start Guide - Fraud Detection Dashboard

## Start the Application

```bash
cd "d:\Zindi Test\project_defcone"
streamlit run streamlit_app.py
```

**Application URL**: http://localhost:8501

---

## 📋 Quick Navigation

### Page 1: Overview
**What you'll see:**
- Total transactions, fraud cases, users, merchants, chargebacks
- Fraud distribution pie chart
- Transaction amount histograms
- Statistical comparison table

**Use this for:** Quick overview and KPIs

### Page 2: EDA (Exploratory Data Analysis)
**6 Analysis Categories:**

1. **Dataset Overview**
   - View any of the 6 datasets
   - Check data types, statistics, missing values
   
2. **Fraud Analysis**
   - Fraud by auth status, channel, payment method
   - 3DS impact on fraud
   - Chargeback analysis

3. **Temporal Analysis**
   - Hourly transaction patterns
   - Day of week trends
   - Daily time series

4. **Geographic Analysis**
   - Top countries by volume
   - Fraud rates by country
   - Country mismatch patterns

5. **Payment Analysis**
   - PSP distribution
   - Decline codes
   - Retry patterns
   - Amount by payment method

6. **Risk Analysis**
   - BIN, IP, Merchant, Email risk distributions
   - User age analysis
   - Correlation with fraud

**Use this for:** Deep dive into patterns and trends

### Page 3: ML Modeling
**Steps to train models:**

1. Review the feature engineering information
2. Adjust parameters:
   - Test set size (default: 30%)
   - Random state (default: 42)
   - Enable feature scaling (recommended)
3. Select models to train (pick 2-5):
   - ✅ Random Forest (recommended)
   - ✅ Gradient Boosting (recommended)
   - Logistic Regression
   - Decision Tree
   - AdaBoost
4. Click **"🚀 Train Models"**
5. Review performance comparison
6. Select a model for detailed analysis

**Use this for:** Model development and evaluation

---

## 💡 Usage Tips

### For First-Time Users
1. Start with **Overview** page (5 min)
2. Explore **EDA** → Fraud Analysis (10 min)
3. Try **ML Modeling** with default settings (5 min)

### For Data Analysts
1. Use **EDA** → Dataset Overview to understand data
2. Run all 6 EDA analysis types
3. Export insights by taking screenshots

### For Data Scientists
1. Review feature engineering in ML page
2. Train multiple models for comparison
3. Analyze feature importance
4. Check ROC curves and confusion matrices

### For Business Users
1. Focus on **Overview** page metrics
2. Use **EDA** → Fraud Analysis for patterns
3. Check **Geographic Analysis** for regional trends

---

## 🎯 Common Tasks

### Find High-Risk Countries
1. Go to **EDA** page
2. Select **Geographic Analysis**
3. View "Fraud Rate by Country" chart

### Identify Best ML Model
1. Go to **ML Modeling** page
2. Select all 5 models
3. Enable feature scaling
4. Click "Train Models"
5. Compare ROC-AUC scores in table

### Analyze Temporal Patterns
1. Go to **EDA** page
2. Select **Temporal Analysis**
3. Review hourly and daily trends

### Check Feature Importance
1. Train models in **ML Modeling** page
2. Scroll to "Detailed Model Analysis"
3. Select a model from dropdown
4. View feature importance chart

---

## 📊 What Each Metric Means

### Overview Page Metrics
- **Total Transactions**: Count of all transactions in dataset
- **Fraud Cases**: Number of fraudulent transactions
- **Fraud Rate (%)**: Percentage of fraud in total transactions
- **Total Users**: Unique users who made transactions
- **Total Merchants**: Unique merchants in transactions
- **Chargebacks**: Number of disputed transactions

### ML Performance Metrics
- **Accuracy**: Overall correctness (correct predictions / total)
- **Precision**: Of predicted frauds, how many were correct
- **Recall**: Of actual frauds, how many were caught
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish classes (0.5-1.0)

**Best for Fraud Detection:** High Recall (catch more frauds) with acceptable Precision

---

## 🔧 Troubleshooting

### Application won't start
```bash
# Install requirements
pip install -r requirements.txt

# Try again
streamlit run streamlit_app.py
```

### Data not loading
- Ensure you're in the correct directory
- Check that `Data/Fraud/` folder exists
- Verify CSV files are present

### Charts not showing
- Check browser console for errors
- Try refreshing the page (F5)
- Clear Streamlit cache (Settings → Clear Cache)

### Model training is slow
- Select fewer models (1-2 instead of 5)
- This is normal for 120K+ transactions
- Wait 10-30 seconds for completion

---

## ⌨️ Keyboard Shortcuts

- **R**: Rerun the application
- **C**: Clear cache
- **Ctrl + Shift + R**: Hard refresh
- **F11**: Full screen mode

---

## 📱 Best Practices

### Performance
- Use **wide layout** (already set)
- Don't train all models repeatedly (use cached results)
- Close browser tab when done to free memory

### Analysis
- Start broad (Overview) → go specific (EDA categories)
- Compare multiple models before choosing one
- Look for patterns across multiple dimensions

### Interpretation
- High fraud rate in a segment doesn't mean causation
- Consider sample size (countries with 100+ transactions)
- Cross-reference findings across multiple charts

---

## 📈 Example Workflow

**Goal: Reduce fraud by 20%**

1. **Identify Problem Areas** (Overview + EDA)
   - Check current fraud rate
   - Find high-risk countries/channels
   - Identify peak fraud times

2. **Understand Patterns** (EDA Deep Dive)
   - Run Fraud Analysis
   - Check Geographic patterns
   - Analyze Payment methods
   - Review Risk factors

3. **Build Predictive Model** (ML Modeling)
   - Train multiple models
   - Select best performer (highest ROC-AUC)
   - Review feature importance
   - Note top fraud indicators

4. **Take Action**
   - Flag high-risk transactions
   - Implement additional checks for risky features
   - Monitor selected countries/channels more closely
   - Require 3DS for high-risk transactions

5. **Monitor Results**
   - Track fraud rate changes
   - Retrain models with new data
   - Adjust thresholds as needed

---

## 🎓 Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Scikit-learn Guide**: https://scikit-learn.org
- **Plotly Charts**: https://plotly.com/python/

## 📝 Data Files Used

All data is from `Data/Fraud/` folder:
- ✅ transactions_raw.csv (main)
- ✅ users.csv
- ✅ devices.csv
- ✅ ips.csv
- ✅ merchants.csv
- ✅ chargebacks.csv

---

**Need more help?** Check [FRAUD_APP_GUIDE.md](FRAUD_APP_GUIDE.md) for detailed documentation.

**Happy Analyzing! 🔍**
