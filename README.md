🛒 Shopper Spectrum: E-Commerce Customer Analytics
A comprehensive e-commerce analytics platform that provides customer segmentation and product recommendations using machine learning.
🚀 Live Demo
Try the live application here
📋 Features
🎯 Product Recommendations

Item-based collaborative filtering
Real-time product similarity analysis
Smart product search functionality
Fallback recommendations for new products

👥 Customer Segmentation

RFM (Recency, Frequency, Monetary) analysis
K-Means clustering for customer groups
Real-time segment prediction
Personalized marketing recommendations

📊 Analytics Dashboard

Interactive customer behavior visualizations
Revenue analysis by segments
Product performance metrics
Business intelligence insights

🛠️ Technology Stack

Frontend: Streamlit
Data Processing: Pandas, NumPy
Machine Learning: Scikit-learn
Visualization: Plotly, Matplotlib, Seaborn
Deployment: Streamlit Cloud

📁 Project Structure
shopper-spectrum/
├── streamlit_app/
│   └── app.py                 # Main Streamlit application
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_recommendation_system.ipynb
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Cleaned and processed data
├── models/                    # Trained ML models
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
🏃‍♂️ Quick Start
Local Installation

Clone the repository
bashgit clone https://github.com/Incharapm0915/shopper-spectrum.git
cd shopper-spectrum

Create virtual environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bashpip install -r requirements.txt

Run the application
bashstreamlit run streamlit_app/app.py

Open your browser to http://localhost:8501

📊 Dataset
The project uses e-commerce transaction data containing:

Customer purchase history
Product information
Transaction details
Geographic data

🤖 Machine Learning Models
Customer Segmentation

Algorithm: K-Means Clustering
Features: RFM (Recency, Frequency, Monetary) metrics
Segments: Champions, Loyal Customers, At Risk, New Customers, etc.

Recommendation System

Algorithm: Item-based Collaborative Filtering
Similarity Metric: Cosine Similarity
Features: Customer-Product interaction matrix

📈 Business Impact

Customer Insights: Identify high-value customers and at-risk segments
Personalized Marketing: Target customers with relevant campaigns
Product Recommendations: Increase cross-selling and upselling
Revenue Optimization: Data-driven business decisions

🔧 Configuration
The application automatically loads pre-trained models and processed data. To retrain models:

Run the Jupyter notebooks in sequence (01-05)
Ensure all model files are saved in the models/ directory
Restart the Streamlit application

📝 Usage Examples
Product Recommendations
python# Search for products containing "WHITE"
# Get 5 similar product recommendations
# View similarity scores and customer metrics
Customer Segmentation
python# Input: Recency=30, Frequency=5, Monetary=500
# Output: "Loyal Customers" segment
# Receive personalized marketing recommendations
🚀 Deployment
The application is deployed on Streamlit Cloud for easy access and sharing.
Local Development
bashstreamlit run streamlit_app/app.py
Production Deployment
Automatically deployed via GitHub integration with Streamlit Cloud.
🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
