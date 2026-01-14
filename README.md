# 🏥 Healthcare Cost Analysis & Prediction Dashboard

An interactive dashboard analyzing healthcare insurance costs with machine learning prediction capabilities. Built to identify key cost drivers and enable accurate healthcare cost estimation.

##  Project Overview

This project analyzes 1,338 patient records to understand healthcare cost patterns and predict insurance charges based on patient characteristics. The analysis reveals that **smoking status combined with BMI is the dominant predictor** of healthcare costs, accounting for 77% of the model's predictive power.

##  Key Findings

- **Smokers incur 380% higher healthcare costs** ($32,050 vs $8,434 annually)
- **87% prediction accuracy** (R² score of 0.87) using Random Forest model
- **BMI×Smoking interaction** is the strongest cost predictor (77% feature importance)
- **89.9% of predictions within $5,000** of actual costs

##  Features

### Interactive Dashboard (4 Tabs)
1. **Overview**: Key metrics, cost distributions, regional analysis
2. **Deep Dive**: Filtered analysis with interactive visualizations
3. **Cost Predictor**: Real-time cost prediction based on patient inputs
4. **Model Performance**: Feature importance and model metrics

### Technical Capabilities
- Predictive modeling with Random Forest Regressor
- Feature engineering (interaction terms, encoding)
- Interactive visualizations with Plotly
- Real-time predictions via Streamlit interface

##  Tech Stack

- **Python 3.x**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Development**: Jupyter Notebooks

##  Project Structure
```
healthcare-cost-analysis-dashboard/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and engineered features
├── src/
│   └── dashboard.py            # Main Streamlit dashboard
├── notebooks/
│   └── 01_insurance_data_exploration.ipynb
├── requirements.txt
└── README.md
```

##  Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthcare-cost-analysis-dashboard.git
cd healthcare-cost-analysis-dashboard
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the dashboard**
```bash
streamlit run src/dashboard.py
```

5. **Open in browser**
Dashboard will automatically open at `http://localhost:8501`

##  Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.8709 |
| Mean Absolute Error | $2,433.71 |
| RMSE | $4,476.13 |
| Accuracy within $5k | 89.9% |
| Accuracy within $10k | 93.3% |

##  Business Insights

1. **Risk Assessment**: Smoking status should be weighted heavily in insurance risk models
2. **Preventive Care**: Smoking cessation programs could reduce costs by ~$23,600 per patient
3. **Pricing Strategy**: BMI + smoking interaction enables more accurate premium calculation
4. **Regional Factors**: Geographic location has minimal impact on costs (<5% variance)

##  Dataset

**Source**: Medical Cost Personal Dataset (Kaggle)
- **Records**: 1,338 patients
- **Features**: Age, Sex, BMI, Children, Smoker Status, Region
- **Target**: Annual healthcare insurance charges ($)
- **Period**: Historical insurance billing data

##  Future Enhancements

- [ ] Add larger hospital discharge dataset (100k+ records)
- [ ] Implement additional ML models (XGBoost, Neural Networks)
- [ ] Add demographic trend analysis
- [ ] Include procedure-level cost breakdown
- [ ] Deploy to cloud (AWS/Heroku)

##  Author

**Spandan** - Data Science Graduate Student, Northeastern University
- 🔗 [LinkedIn](https://www.linkedin.com/in/s-spandan)
- 🌐 [Portfolio](https://spandansurdas.vercel.app/)
- 📧 Email: spandan.surdas25@.gmail@example.com

##  License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: Kaggle Medical Cost Personal Dataset
- Previous experience: Antimicrobial resistance prediction at Medtigo
- Research background: Disease-mapping ML models (89% accuracy)