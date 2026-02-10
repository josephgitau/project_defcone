# 🔍 Fraud Detection Dashboard - Project Defcone

A comprehensive Streamlit web application for fraud detection analysis and machine learning modeling.

## 🌟 Features

### 📊 Interactive Dashboards
- **Overview Dashboard**: High-level metrics and KPIs
- **EDA Module**: 6 specialized analysis categories
- **ML Modeling**: Multiple algorithm training and comparison

### 🎯 Analysis Capabilities
- Fraud pattern identification across multiple dimensions
- Temporal and geographic fraud analysis
- Payment method and PSP risk assessment
- Risk factor correlation analysis
- Chargeback investigation

### 🤖 Machine Learning
- 5 classification algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost)
- Automated feature engineering
- Model performance comparison
- Feature importance analysis
- Interactive ROC and Precision-Recall curves

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
cd project_defcone

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run streamlit_app.py
```

Access the dashboard at: **http://localhost:8501**

## 📁 Project Structure

```
project_defcone/
├── streamlit_app.py          # Main application file
├── requirements.txt           # Python dependencies
├── FRAUD_APP_GUIDE.md        # Comprehensive user guide
├── README.md                  # This file
└── Data/
    └── Fraud/
        ├── transactions_raw.csv    # Main transaction data (120K+ records)
        ├── users.csv               # User information (30K+ users)
        ├── devices.csv             # Device fraud hints (18K+ devices)
        ├── ips.csv                 # IP risk levels (45K+ IPs)
        ├── merchants.csv           # Merchant risk scores (2K+ merchants)
        └── chargebacks.csv         # Chargeback details (20K+ records)
```

## 📊 Datasets Overview

| Dataset | Records | Key Information |
|---------|---------|-----------------|
| Transactions | 120,000+ | Main data with fraud labels, amounts, risk scores |
| Users | 30,000+ | User age, email risk |
| Devices | 18,000+ | Device fraud hints |
| IPs | 45,000+ | IP risk levels |
| Merchants | 2,000+ | Merchant risk scores |
| Chargebacks | 20,000+ | Chargeback reasons and timing |

## 🎨 Application Pages

### 1. Overview Page
- Key performance metrics
- Dataset statistics
- Fraud distribution visualization
- Transaction amount analysis

### 2. EDA Page
Choose from 6 analysis types:
- **Dataset Overview**: Data preview, types, statistics, missing values
- **Fraud Analysis**: Auth status, channels, payment methods, 3DS impact
- **Temporal Analysis**: Hourly, daily, weekly patterns
- **Geographic Analysis**: Country-based fraud patterns
- **Payment Analysis**: PSP, decline codes, retry patterns
- **Risk Analysis**: BIN, IP, merchant, email risk distributions

### 3. ML Modeling Page
- Feature engineering pipeline
- Multi-model training
- Performance comparison dashboard
- Detailed model evaluation
- Feature importance visualization

## 🔧 Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Machine Learning**: Scikit-learn
- **Python**: 3.8+

## 📈 Key Metrics & Insights

The application tracks and visualizes:
- Fraud rate and distribution
- Transaction volumes by various dimensions
- Risk score distributions
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance for fraud prediction

## 🎯 Use Cases

1. **Fraud Investigation**: Analyze patterns and anomalies
2. **Risk Assessment**: Evaluate transaction risk factors
3. **Model Development**: Train and compare ML models
4. **Business Intelligence**: Track KPIs and trends

## 📖 Documentation

For detailed usage instructions, see [FRAUD_APP_GUIDE.md](FRAUD_APP_GUIDE.md)

## 🔒 Data Privacy

This application uses sample fraud detection data for demonstration purposes. Ensure proper data handling and privacy measures when using with real data.

## 🚀 Future Enhancements

- [ ] Real-time prediction API
- [ ] Model deployment pipeline
- [ ] SHAP value integration
- [ ] Custom threshold optimization
- [ ] Automated model retraining
- [ ] Alert system for high-risk transactions

## 📝 Requirements

```
streamlit
pandas
numpy
seaborn
matplotlib
scikit-learn
plotly
```

## 🤝 Contributing

This is a starter template for ML model deployment and fraud detection analysis.

## 📄 License

See LICENSE file for details.

---

**Built with ❤️ using Streamlit**
