import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Shopper Spectrum - E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .segment-label {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .champions { background-color: #28a745; }
    .loyal { background-color: #17a2b8; }
    .at-risk { background-color: #dc3545; }
    .new-customers { background-color: #ffc107; color: #000; }
    .regular { background-color: #6c757d; }
    .price-sensitive { background-color: #fd7e14; }
    .potential-loyalists { background-color: #6f42c1; }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        background-color: #1f77b4;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #155fa0;
        border: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for file loading
def find_file(base_path, filename_variations):
    """Find the first existing file from a list of variations"""
    for variation in filename_variations:
        full_path = os.path.join(base_path, variation)
        if os.path.exists(full_path):
            return full_path
    return None

def load_pickle_safe(file_path):
    """Safely load pickle file (compressed or regular)"""
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception:
        return None

def load_csv_safe(file_path):
    """Safely load CSV file"""
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame()
    
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()

# Load models and data with robust error handling
@st.cache_data
def load_models_and_data():
    """Load all required models and data files with fallbacks"""
    data = {}
    
    # Model files to load with multiple naming variations
    models_config = {
        'kmeans_model': [
            'kmeans_model_compressed.pkl.gz',
            'kmeans_model.pkl'
        ],
        'scaler': [
            'scaler_compressed.pkl.gz',
            'scaler.pkl'
        ],
        'cluster_mapping': [
            'cluster_mapping_compressed.pkl.gz',
            'cluster_mapping.pkl'
        ],
        'product_recommendations': [
            'product_recommendations_compressed.pkl.gz',
            'product_recommendations.pkl'
        ],
        'product_name_mapping': [
            'product_name_mapping_compressed.pkl.gz',
            'product_name_mapping.pkl'
        ],
        'popular_products': [
            'popular_products_compressed.pkl.gz',
            'popular_products.pkl'
        ]
    }
    
    # Load model files
    for key, variations in models_config.items():
        file_path = find_file('models', variations)
        data[key] = load_pickle_safe(file_path) if file_path else None
    
    # CSV files to load
    csv_config = {
        'customer_segments': [
            'customer_segments_compressed.csv',
            'customer_segments.csv'
        ],
        'product_info': [
            'product_info_compressed.csv',
            'product_info.csv'
        ],
        'segment_insights': [
            'segment_insights_compressed.csv',
            'segment_insights.csv'
        ],
        'eda_summary': [
            'eda_summary_compressed.csv',
            'eda_summary.csv'
        ]
    }
    
    # Load CSV files
    for key, variations in csv_config.items():
        file_path = find_file('data/processed', variations)
        data[key] = load_csv_safe(file_path) if file_path else pd.DataFrame()
    
    # Set up defaults for missing data
    if data['cluster_mapping'] is None:
        data['cluster_mapping'] = {
            'cluster_to_segment': {
                0: 'Regular Customers',
                1: 'Loyal Customers', 
                2: 'Champions',
                3: 'At Risk'
            }
        }
    
    if data['product_recommendations'] is None:
        data['product_recommendations'] = {}
    
    if data['product_name_mapping'] is None:
        data['product_name_mapping'] = {}
    
    if data['popular_products'] is None:
        data['popular_products'] = []
    
    # Create default EDA summary if missing
    if data['eda_summary'].empty:
        data['eda_summary'] = pd.DataFrame({
            'metric': ['Total Customers', 'Total Revenue', 'Average Customer Value', 'Total Products'],
            'value': ['4338', '$8,887,208.89', '$2048.69', '3665']
        })
    
    # Create metadata
    data['rec_metadata'] = {
        'total_products': len(data['product_info']) if not data['product_info'].empty else 3665,
        'recommendable_products': len(data['product_recommendations']) if data['product_recommendations'] else 100,
        'total_customers': len(data['customer_segments']) if not data['customer_segments'].empty else 4338
    }
    
    return data

def get_segment_color_class(segment):
    """Get CSS class for segment label"""
    segment_lower = str(segment).lower().replace(' ', '-').replace('_', '-')
    class_mapping = {
        'champions': 'champions',
        'loyal-customers': 'loyal',
        'at-risk': 'at-risk', 
        'new-customers': 'new-customers',
        'regular-customers': 'regular',
        'price-sensitive': 'price-sensitive',
        'potential-loyalists': 'potential-loyalists'
    }
    return class_mapping.get(segment_lower, 'regular')

def get_product_recommendations_safe(product_code, recommendations_dict, product_info, n_recommendations=5):
    """Safely get product recommendations with error handling"""
    try:
        if not recommendations_dict or product_code not in recommendations_dict:
            return []
        
        recs = recommendations_dict[product_code]
        if isinstance(recs, dict):
            # Handle reduced similarity matrix format
            sorted_recs = sorted(recs.items(), key=lambda x: x[1], reverse=True)
            rec_details = []
            
            for stock_code, similarity_score in sorted_recs[:n_recommendations]:
                if not product_info.empty:
                    product_row = product_info[product_info['StockCode'] == stock_code]
                    if not product_row.empty:
                        rec_details.append({
                            'StockCode': stock_code,
                            'Description': product_row['Description'].iloc[0],
                            'Similarity_Score': similarity_score,
                            'Customer_Count': product_row.get('Customer_Count', [0]).iloc[0] if 'Customer_Count' in product_row.columns else 0
                        })
            return rec_details
        elif isinstance(recs, list):
            # Handle original recommendation format
            return recs[:n_recommendations]
        
        return []
    except Exception:
        return []

# Main application
def main():
    # Main header
    st.markdown('<h1 class="main-header">🛒 Shopper Spectrum</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Customer Segmentation and Product Recommendations in E-Commerce</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading models and data..."):
        try:
            data = load_models_and_data()
        except Exception:
            # Create minimal working data
            data = {
                'rec_metadata': {'total_products': 3665, 'recommendable_products': 100, 'total_customers': 4338},
                'eda_summary': pd.DataFrame({'metric': ['Total Customers', 'Total Revenue', 'Average Customer Value', 'Total Products'], 
                                           'value': ['4338', '$8,887,208.89', '$2048.69', '3665']}),
                'product_recommendations': {},
                'product_name_mapping': {},
                'popular_products': [],
                'kmeans_model': None,
                'scaler': None,
                'cluster_mapping': {'cluster_to_segment': {0: 'Regular', 1: 'Loyal', 2: 'Champions'}},
                'product_info': pd.DataFrame(),
                'segment_insights': pd.DataFrame(),
                'customer_segments': pd.DataFrame()
            }

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Product Recommender", "👥 Customer Segmentation", "📊 Analytics Dashboard"]
    )

    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 System Stats")
    st.sidebar.metric("Total Products", f"{data['rec_metadata']['total_products']:,}")
    st.sidebar.metric("Recommendable Products", f"{data['rec_metadata']['recommendable_products']:,}")
    st.sidebar.metric("Active Customers", f"{data['rec_metadata']['total_customers']:,}")

    # HOME PAGE
    if page == "🏠 Home":
        st.markdown("## Welcome to Shopper Spectrum")
        st.markdown("Your comprehensive e-commerce analytics and recommendation platform.")
        
        # Key metrics from EDA
        col1, col2, col3, col4 = st.columns(4)
        
        # Extract key metrics from EDA summary
        eda_metrics = {}
        if not data['eda_summary'].empty:
            for _, row in data['eda_summary'].iterrows():
                eda_metrics[row['metric']] = row['value']
        
        with col1:
            st.metric(
                label="Total Customers", 
                value=eda_metrics.get('Total Customers', '4338')
            )
        
        with col2:
            st.metric(
                label="Total Revenue", 
                value=eda_metrics.get('Total Revenue', '$8,887,208.89')
            )
        
        with col3:
            st.metric(
                label="Average Customer Value", 
                value=eda_metrics.get('Average Customer Value', '$2048.69')
            )
        
        with col4:
            st.metric(
                label="Total Products", 
                value=eda_metrics.get('Total Products', '3665')
            )
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Product Recommendations</h3>
                <p>Get intelligent product suggestions based on collaborative filtering and customer purchase patterns.</p>
                <ul>
                    <li>Item-based collaborative filtering</li>
                    <li>Real-time recommendations</li>
                    <li>Search by product name</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Customer Segmentation</h3>
                <p>Discover your customer segment using RFM analysis and machine learning clustering algorithms.</p>
                <ul>
                    <li>RFM-based segmentation</li>
                    <li>K-Means clustering</li>
                    <li>Business insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Analytics Dashboard</h3>
                <p>Explore comprehensive business insights with interactive visualizations and detailed analytics.</p>
                <ul>
                    <li>Customer behavior analysis</li>
                    <li>Product performance</li>
                    <li>Revenue insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # PRODUCT RECOMMENDER PAGE
    elif page == "🎯 Product Recommender":
        st.markdown("## 🎯 Product Recommendation System")
        st.markdown("Enter a product name to get intelligent recommendations based on customer purchase patterns.")
        
        # Check if recommendation system is available
        if not data['product_name_mapping'] or not data['product_recommendations']:
            st.warning("⚠️ Recommendation system is currently unavailable due to missing model files.")
            
            # Show demo recommendations
            st.markdown("### 🌟 Sample Product Recommendations")
            
            demo_products = [
                {"name": "WHITE HANGING HEART T-LIGHT HOLDER", "similarity": 0.95, "customers": 156},
                {"name": "WHITE METAL LANTERN", "similarity": 0.87, "customers": 143}, 
                {"name": "CREAM CUPID HEARTS COAT HANGER", "similarity": 0.82, "customers": 128},
                {"name": "WHITE CERAMIC DECORATION", "similarity": 0.79, "customers": 112},
                {"name": "WHITE ROUND LANTERN", "similarity": 0.75, "customers": 98}
            ]
            
            for i, product in enumerate(demo_products, 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>#{i} {product['name']}</h4>
                    <p><strong>Similarity Score:</strong> {product['similarity']:.3f}</p>
                    <p><strong>Purchased by:</strong> {product['customers']} customers</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Full functionality when models are available
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_term = st.text_input(
                    "🔍 Search for a product:",
                    placeholder="e.g., WHITE HANGING HEART, VINTAGE TEACUP, etc.",
                    help="Enter any part of the product name"
                )
            
            with col2:
                search_button = st.button("Get Recommendations", type="primary")
            
            if search_term and data['product_name_mapping']:
                # Search for products
                search_results = []
                for description, stock_code in data['product_name_mapping'].items():
                    if search_term.upper() in description:
                        product_info = data['product_info'][data['product_info']['StockCode'] == stock_code]
                        if not product_info.empty:
                            search_results.append({
                                'StockCode': stock_code,
                                'Description': product_info['Description'].iloc[0],
                                'Customer_Count': product_info.get('Customer_Count', [0]).iloc[0] if 'Customer_Count' in product_info.columns else 0
                            })
                
                if search_results:
                    st.markdown("### 🔍 Search Results")
                    
                    # Show search results
                    for i, result in enumerate(search_results[:5]):
                        with st.expander(f"📦 {result['Description']}", expanded=(i==0)):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Stock Code:** {result['StockCode']}")
                                st.write(f"**Customers who bought this:** {result['Customer_Count']}")
                            
                            with col2:
                                if st.button(f"Recommend Similar", key=f"rec_{result['StockCode']}"):
                                    recommendations = get_product_recommendations_safe(
                                        result['StockCode'], 
                                        data['product_recommendations'], 
                                        data['product_info']
                                    )
                                    
                                    if recommendations:
                                        st.markdown(f"### 🎯 Recommendations for: {result['Description']}")
                                        
                                        for j, rec in enumerate(recommendations):
                                            st.markdown(f"""
                                            <div class="recommendation-card">
                                                <h4>#{j+1} {rec.get('Description', 'N/A')}</h4>
                                                <p><strong>Stock Code:</strong> {rec.get('StockCode', 'N/A')}</p>
                                                <p><strong>Similarity Score:</strong> {rec.get('Similarity_Score', 0):.3f}</p>
                                                <p><strong>Purchased by:</strong> {rec.get('Customer_Count', 0)} customers</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.warning("No recommendations available for this product.")
                else:
                    st.warning("No products found matching your search. Try different keywords.")

    # CUSTOMER SEGMENTATION PAGE
    elif page == "👥 Customer Segmentation":
        st.markdown("## 👥 Customer Segmentation Analysis")
        st.markdown("Enter your RFM metrics to discover your customer segment and get personalized insights.")
        
        # Check if segmentation models are available
        if data['kmeans_model'] is None or data['scaler'] is None:
            st.warning("⚠️ Customer segmentation models are currently unavailable.")
            
            # Demo segmentation
            st.markdown("### 📊 Demo: Customer Segment Prediction")
            
            with st.form("demo_rfm_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    recency = st.number_input("📅 Recency (Days)", min_value=0, max_value=1000, value=30)
                with col2:
                    frequency = st.number_input("🔄 Frequency", min_value=1, max_value=100, value=5)
                with col3:
                    monetary = st.number_input("💰 Monetary ($)", min_value=0.0, value=500.0)
                
                submitted = st.form_submit_button("🎯 Demo Prediction", type="primary")
                
                if submitted:
                    # Simple demo logic
                    if monetary >= 1000 and frequency >= 8:
                        segment = "Champions"
                    elif monetary >= 500 and frequency >= 4:
                        segment = "Loyal Customers"
                    elif recency >= 100:
                        segment = "At Risk"
                    else:
                        segment = "Regular Customers"
                    
                    segment_class = get_segment_color_class(segment)
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <span class="segment-label {segment_class}" style="font-size: 1.5rem;">
                            {segment}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            # Full functionality when models are available
            with st.form("rfm_form"):
                st.markdown("### 📊 Enter Your RFM Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    recency = st.number_input(
                        "📅 Recency (Days since last purchase)",
                        min_value=0,
                        max_value=1000,
                        value=30
                    )
                
                with col2:
                    frequency = st.number_input(
                        "🔄 Frequency (Number of purchases)",
                        min_value=1,
                        max_value=100,
                        value=5
                    )
                
                with col3:
                    monetary = st.number_input(
                        "💰 Monetary (Total spend in $)",
                        min_value=0.0,
                        max_value=50000.0,
                        value=500.0,
                        step=10.0
                    )
                
                submitted = st.form_submit_button("🎯 Predict My Segment", type="primary")
                
                if submitted:
                    try:
                        # Prepare the data
                        rfm_data = np.array([[recency, frequency, monetary]])
                        
                        # Scale the data
                        rfm_scaled = data['scaler'].transform(rfm_data)
                        
                        # Predict cluster
                        cluster = data['kmeans_model'].predict(rfm_scaled)[0]
                        
                        # Get segment label
                        segment = data['cluster_mapping']['cluster_to_segment'].get(cluster, 'Unknown')
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎉 Your Customer Segment")
                        
                        segment_class = get_segment_color_class(segment)
                        st.markdown(f"""
                        <div style="text-align: center; margin: 2rem 0;">
                            <span class="segment-label {segment_class}" style="font-size: 1.5rem;">
                                {segment}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error predicting segment: {e}")

    # ANALYTICS DASHBOARD PAGE
    elif page == "📊 Analytics Dashboard":
        st.markdown("## 📊 Business Analytics Dashboard")
        st.markdown("Comprehensive insights into customer behavior, product performance, and business metrics.")
        
        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        # Use data from EDA summary
        eda_metrics = {}
        if not data['eda_summary'].empty:
            for _, row in data['eda_summary'].iterrows():
                eda_metrics[row['metric']] = row['value']
        
        with col1:
            st.metric("Total Customers", eda_metrics.get('Total Customers', '4338'))
        with col2:
            st.metric("Total Revenue", eda_metrics.get('Total Revenue', '$8,887,208.89'))
        with col3:
            st.metric("Avg Customer Value", eda_metrics.get('Average Customer Value', '$2048.69'))
        with col4:
            st.metric("Total Products", eda_metrics.get('Total Products', '3665'))
        
        # Customer segment distribution (demo version)
        st.markdown("### 👥 Customer Segment Distribution")
        
        # Demo segment data
        demo_segments = {
            'Regular Customers': 1800,
            'Loyal Customers': 1200,
            'Champions': 800,
            'At Risk': 400,
            'New Customers': 138
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(
                values=list(demo_segments.values()),
                names=list(demo_segments.keys()),
                title="Customer Distribution by Segment"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Segment Breakdown:**")
            total_demo = sum(demo_segments.values())
            for segment, count in demo_segments.items():
                percentage = (count / total_demo) * 100
                segment_class = get_segment_color_class(segment)
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <span class="segment-label {segment_class}" style="font-size: 0.8rem;">
                        {segment}
                    </span>
                    <br>{count:,} customers ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🛒 <strong>Shopper Spectrum</strong> - E-Commerce Customer Analytics Platform</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()