# 🔍 Fraud Detection Dashboard - User Guide

## Overview
This comprehensive Streamlit web application provides end-to-end fraud detection analysis and machine learning modeling capabilities using fraud transaction data.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run streamlit_app.py
```

The application will be available at: `http://localhost:8501`

## 📊 Application Structure

### 1. **Overview Page**
The landing page provides high-level insights into the fraud detection datasets:

#### Key Metrics Dashboard
- **Total Transactions**: Complete count of all transactions
- **Fraud Cases**: Number and percentage of fraudulent transactions
- **Total Users**: Unique user count
- **Total Merchants**: Unique merchant count
- **Chargebacks**: Total chargeback cases

#### Visualizations
- **Dataset Overview Table**: Summary of all datasets with record counts
- **Fraud Distribution Pie Chart**: Visual representation of fraud vs legitimate transactions
- **Transaction Amount Distribution**: Histogram comparing amounts for fraud vs legitimate transactions
- **Amount Statistics**: Mean, median, and standard deviation by transaction type

### 2. **EDA (Exploratory Data Analysis) Page**
Comprehensive data exploration with six analysis categories:

#### A. Dataset Overview
- **Interactive Dataset Selector**: Choose from 7 datasets (Transactions, Users, Devices, IPs, Merchants, Chargebacks, Merged Data)
- **Sample Data Preview**: First 10 rows of selected dataset
- **Data Types & Non-Null Counts**: Column information table
- **Statistical Summary**: Descriptive statistics for numerical columns
- **Missing Values Visualization**: Bar chart showing missing value counts by column

#### B. Fraud Analysis
- **Fraud by Auth Status**: Grouped bar chart comparing approved/declined transactions
- **Fraud by Channel**: Distribution across Web/App channels
- **Fraud by Payment Method**: Pie chart of fraud distribution across VISA/MASTERCARD
- **3DS Impact**: Analysis of 3D Secure authentication effect on fraud
- **Chargeback Analysis**: 
  - Chargebacks by transaction type
  - Chargeback reasons distribution

#### C. Temporal Analysis
- **Hourly Trends**: Line chart showing transaction volume by hour of day
- **Day of Week Patterns**: Bar chart of transactions by weekday
- **Daily Trend**: Area chart showing daily transaction volume over time
- Identifies peak fraud times and patterns

#### D. Geographic Analysis
- **Top 15 Countries**: Stacked bar chart of transaction volume by country
- **Fraud Rate by Country**: Analysis of fraud rates for countries with 100+ transactions
- **Country Mismatch Analysis**: Impact of transaction country vs card country mismatch on fraud

#### E. Payment Analysis
- **PSP Distribution**: Transaction volume by Payment Service Provider
- **Top 10 Decline Codes**: Most common transaction decline reasons
- **Retry Analysis**: Fraud patterns by retry count
- **Amount by Payment Method**: Box plot comparing transaction amounts

#### F. Risk Analysis
- **BIN Risk Distribution**: Histogram of BIN risk scores
- **IP Risk Distribution**: Histogram of IP risk levels
- **Merchant Risk Distribution**: Risk score analysis for merchants
- **Email Risk Analysis**: High vs low email risk comparison
- **User Age Analysis**: Account age distribution for fraud vs legitimate
- **Correlation Heatmap**: Feature correlation with fraud label

### 3. **ML Modeling Page**
Full machine learning pipeline with model training and evaluation:

#### Feature Engineering
Automated preprocessing includes:
1. DateTime conversion to numerical features (hour, day_of_week, day_of_month)
2. Interaction feature creation (country_mismatch indicator)
3. Categorical variable encoding (channel, psp_name, payment_method, etc.)
4. Missing value handling
5. Optional feature scaling (StandardScaler)

#### Model Configuration
**Adjustable Parameters:**
- Test set size (10%-40%, default: 30%)
- Random state for reproducibility
- Feature scaling toggle
- Multiple model selection:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - AdaBoost

#### Model Performance Comparison
After training, the dashboard displays:
1. **Performance Metrics Table**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
   - Color-coded heatmap highlighting best performers

2. **Grouped Bar Chart**: Side-by-side comparison of all metrics across models

3. **ROC Curves**: Interactive plot with AUC scores for each model

#### Detailed Model Analysis
Select any trained model to view:

1. **Confusion Matrix**: Interactive heatmap showing true/false positives and negatives

2. **Classification Report**: Detailed table with precision, recall, F1-score by class

3. **Precision-Recall Curve**: Trade-off visualization for optimal threshold selection

4. **Feature Importance/Coefficients**:
   - Tree-based models: Feature importance scores
   - Linear models: Coefficient values
   - Top 20 features displayed in horizontal bar chart

## 📈 Key Features

### Data Integration
- Automatically merges 6 fraud datasets:
  - `transactions_raw.csv` (main dataset)
  - `users.csv` (user information)
  - `devices.csv` (device fraud hints)
  - `ips.csv` (IP risk levels)
  - `merchants.csv` (merchant risk scores)
  - `chargebacks.csv` (chargeback information)

### Interactive Visualizations
- **Plotly**: Interactive charts with zoom, pan, and hover tooltips
- **Color Consistency**: Fraud (red) vs Legitimate (green) throughout
- **Responsive Design**: Adapts to screen size

### Performance Optimization
- **Data Caching**: `@st.cache_data` decorator for fast loading
- **Efficient Processing**: Vectorized operations with pandas/numpy

## 🎯 Use Cases

### 1. Fraud Investigation
- Identify high-risk transactions by country, merchant, or user
- Analyze temporal patterns in fraudulent activity
- Review chargeback patterns and reasons

### 2. Risk Assessment
- Evaluate BIN, IP, and merchant risk distributions
- Identify correlations between risk factors and fraud
- Analyze user account age impact on fraud

### 3. Model Development
- Compare multiple ML algorithms
- Fine-tune test/train splits
- Identify most important fraud indicators
- Evaluate model performance with multiple metrics

### 4. Business Intelligence
- Track transaction volumes by channel, PSP, and payment method
- Analyze geographic fraud patterns
- Monitor 3DS authentication effectiveness

## 📊 Featured Datasets

### Main Dataset: transactions_raw.csv (120,000+ transactions)
**Key Fields:**
- `fraud_label`: Target variable (0=legitimate, 1=fraud)
- `amount`: Transaction amount
- `country`, `card_country`: Geographic information
- `channel`: Web or App
- `psp_name`: Payment service provider
- `payment_method`: VISA, MASTERCARD, etc.
- `auth_status`: Approved or Declined
- `bin_risk`, `ip_risk_level`: Risk scores
- `is_3ds`: 3D Secure authentication flag
- `retry_count`: Number of retry attempts

### Supporting Datasets:
- **users.csv**: User age and email risk (30,000+ users)
- **devices.csv**: Device fraud hints (18,000+ devices)
- **ips.csv**: IP risk levels (45,000+ IPs)
- **merchants.csv**: Merchant risk scores (2,000+ merchants)
- **chargebacks.csv**: Chargeback details (20,000+ chargebacks)

## 🔧 Technical Stack

- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models and metrics
- **Seaborn/Matplotlib**: Additional plotting

## 💡 Tips for Best Results

### EDA
1. Start with **Overview** page to understand data distribution
2. Use **Fraud Analysis** to identify key risk factors
3. Check **Temporal Analysis** for time-based patterns
4. Review **Geographic Analysis** for location-based insights

### ML Modeling
1. Enable **Feature Scaling** for better model performance
2. Train multiple models simultaneously for comparison
3. Use **Random Forest** or **Gradient Boosting** for best results
4. Check **Feature Importance** to understand model decisions
5. Analyze **Confusion Matrix** to understand error types
6. Use **ROC-AUC** for imbalanced datasets (fraud detection)

## 🎨 Color Scheme
- **Legitimate Transactions**: Green (#00CC96)
- **Fraudulent Transactions**: Red (#EF553B)
- **Neutral/Other**: Various Plotly color schemes

## 📝 Notes
- All timestamps are in UTC
- Missing values are automatically handled
- Feature scaling is optional but recommended
- Models use stratified train-test split to maintain fraud ratio

## 🚀 Future Enhancements
Potential additions:
- Real-time prediction interface
- Model deployment and API endpoint
- Advanced ensemble methods
- SHAP values for model interpretability
- Custom threshold optimization
- Time series forecasting
- A/B testing framework

## 📞 Support
For issues or questions, refer to the dataset documentation in `Data/readme.md`

---

**Happy Fraud Detection! 🔍**
