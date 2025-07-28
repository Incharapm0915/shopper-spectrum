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

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Helper functions
def find_file(base_path, filename_variations):
    """Find the first existing file from a list of variations"""
    for variation in filename_variations:
        full_path = os.path.join(base_path, variation)
        if os.path.exists(full_path):
            return full_path
    return None

def load_pickle_safe(file_path):
    """Safely load pickle file"""
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

# Load models and data
@st.cache_data
def load_models_and_data():
    """Load all required models and data files"""
    data = {}
    
    # Model files configuration
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
    
    # CSV files configuration
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
    
    # Set defaults for missing data
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
    segment_lower = str(segment).lower().replace(' ', '-')
    class_mapping = {
        'champions': 'champions',
        'loyal-customers': 'loyal',
        'at-risk': 'at-risk',
        'new-customers': 'new-customers',
        'regular-customers': 'regular',
        'price-sensitive': 'price-sensitive'
    }
    return class_mapping.get(segment_lower, 'regular')

def get_recommendations(product_code, recommendations_dict, product_info, n_recommendations=5):
    """Get product recommendations"""
    if not recommendations_dict or product_code not in recommendations_dict:
        return []
    
    try:
        recs = recommendations_dict[product_code]
        if isinstance(recs, list):
            return recs[:n_recommendations]
        elif isinstance(recs, dict):
            # Handle similarity matrix format
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
        return []
    except Exception:
        return []

# Main header
st.markdown('<h1 class="main-header">🛒 Shopper Spectrum</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Customer Segmentation and Product Recommendations in E-Commerce</p>', unsafe_allow_html=True)

# Load data
with st.spinner("Loading models and data..."):
    data = load_models_and_data()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "🎯 Product Recommender", "👥 Customer Segmentation", "📊 Analytics Dashboard"]
)

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 System Stats")
st.sidebar.metric("Total Products", f"{data['rec_metadata']['total_products']:,}")
st.sidebar.metric("Recommendable Products", f"{data['rec_metadata']['recommendable_products']:,}")
st.sidebar.metric("Active Customers", f"{data['rec_metadata']['total_customers']:,}")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("## Welcome to Shopper Spectrum")
    st.markdown("Your comprehensive e-commerce analytics and recommendation platform.")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    eda_metrics = {}
    if not data['eda_summary'].empty:
        for _, row in data['eda_summary'].iterrows():
            eda_metrics[row['metric']] = row['value']
    
    with col1:
        st.metric("Total Customers", eda_metrics.get('Total Customers', '4338'))
    with col2:
        st.metric("Total Revenue", eda_metrics.get('Total Revenue', '$8,887,208.89'))
    with col3:
        st.metric("Average Customer Value", eda_metrics.get('Average Customer Value', '$2048.69'))
    with col4:
        st.metric("Total Products", eda_metrics.get('Total Products', '3665'))
    
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
            <p>Discover your customer segment using RFM analysis and machine learning clustering.</p>
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
            <p>Explore comprehensive business insights with interactive visualizations.</p>
            <ul>
                <li>Customer behavior analysis</li>
                <li>Product performance</li>
                <li>Revenue insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("---")
    st.markdown("## 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### For Product Recommendations:
        1. Go to **Product Recommender** page
        2. Enter a product name in the search box
        3. Get 5 similar product recommendations
        4. Explore related products and insights
        """)
    
    with col2:
        st.markdown("""
        ### For Customer Segmentation:
        1. Go to **Customer Segmentation** page
        2. Enter RFM values (Recency, Frequency, Monetary)
        3. Get your customer segment prediction
        4. View personalized recommendations
        """)

# PRODUCT RECOMMENDER PAGE
elif page == "🎯 Product Recommender":
    st.markdown("## 🎯 Product Recommendation System")
    st.markdown("Enter a product name to get intelligent recommendations based on customer purchase patterns.")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input(
            "🔍 Search for a product:",
            placeholder="e.g., WHITE HANGING HEART, VINTAGE TEACUP, etc.",
            help="Enter any part of the product name"
        )
    
    with col2:
        search_button = st.button("Get Recommendations", type="primary")
    
    # Show recommendations
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
            
            for i, result in enumerate(search_results[:5]):
                with st.expander(f"📦 {result['Description']}", expanded=(i==0)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Stock Code:** {result['StockCode']}")
                        st.write(f"**Customers who bought this:** {result['Customer_Count']}")
                    
                    with col2:
                        if st.button(f"Get Similar Products", key=f"rec_{result['StockCode']}"):
                            recommendations = get_recommendations(
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
            st.warning("No products found matching your search.")
    
    elif search_term:
        st.warning("Product search is currently unavailable. Please check back later.")
    
    # Example searches
    st.markdown("---")
    st.markdown("### 💡 Try These Example Searches:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤍 WHITE ITEMS"):
            st.rerun()
    
    with col2:
        if st.button("🎄 CHRISTMAS"):
            st.rerun()
    
    with col3:
        if st.button("☕ COFFEE"):
            st.rerun()

# CUSTOMER SEGMENTATION PAGE
elif page == "👥 Customer Segmentation":
    st.markdown("## 👥 Customer Segmentation Analysis")
    st.markdown("Enter your RFM metrics to discover your customer segment and get personalized insights.")
    
    # RFM input form
    with st.form("rfm_form"):
        st.markdown("### 📊 Enter Your RFM Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recency = st.number_input(
                "📅 Recency (Days since last purchase)",
                min_value=0,
                max_value=1000,
                value=30,
                help="Number of days since the customer's last purchase"
            )
        
        with col2:
            frequency = st.number_input(
                "🔄 Frequency (Number of purchases)",
                min_value=1,
                max_value=100,
                value=5,
                help="Total number of purchases made by the customer"
            )
        
        with col3:
            monetary = st.number_input(
                "💰 Monetary (Total spend in $)",
                min_value=0.0,
                max_value=50000.0,
                value=500.0,
                step=10.0,
                help="Total amount spent by the customer"
            )
        
        submitted = st.form_submit_button("🎯 Predict My Segment", type="primary")
        
        if submitted:
            if data['kmeans_model'] is not None and data['scaler'] is not None:
                try:
                    # Prepare and scale data
                    rfm_data = np.array([[recency, frequency, monetary]])
                    rfm_scaled = data['scaler'].transform(rfm_data)
                    
                    # Predict cluster
                    cluster = data['kmeans_model'].predict(rfm_scaled)[0]
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
                    
                    # Show segment insights if available
                    if not data['segment_insights'].empty and segment in data['segment_insights'].index:
                        segment_data = data['segment_insights'].loc[segment]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Average Recency", f"{segment_data.get('Avg_Recency', 0):.0f} days")
                        with col2:
                            st.metric("Average Frequency", f"{segment_data.get('Avg_Frequency', 0):.1f} orders")
                        with col3:
                            st.metric("Average Spend", f"${segment_data.get('Avg_Monetary', 0):,.2f}")
                        with col4:
                            st.metric("Customer Count", f"{segment_data.get('Customer_Count', 0):,}")
                    
                except Exception as e:
                    st.error(f"Error predicting segment: {e}")
            else:
                # Demo prediction when models are unavailable
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
                
                st.info("This is a demo prediction based on simple rules. The actual ML model is not available.")
    
    # Segment recommendations
    st.markdown("### 💡 Personalized Recommendations by Segment")
    
    recommendations_map = {
        'Champions': [
            "🏆 You're our most valuable customer! Consider our VIP membership program.",
            "🎁 Get early access to new product launches and exclusive deals.",
            "💎 Explore our premium product collection curated just for you."
        ],
        'Loyal Customers': [
            "🤝 Thank you for your loyalty! Check out our customer rewards program.",
            "📦 Try our subscription service for regular deliveries.",
            "🆙 Explore higher-value products that match your interests."
        ],
        'At Risk': [
            "💔 We miss you! Here's a special 20% discount to welcome you back.",
            "📞 Let us know if there's anything we can improve.",
            "🎯 Check out new products that match your previous purchases."
        ],
        'Regular Customers': [
            "⭐ Thanks for being a regular customer! Try our monthly deals.",
            "🔄 Set up automatic reorders for frequently bought items.",
            "📊 View your purchase history to discover new favorites."
        ]
    }
    
    for segment, recs in recommendations_map.items():
        with st.expander(f"📊 {segment} Recommendations"):
            for i, rec in enumerate(recs, 1):
                st.markdown(f"{i}. {rec}")

# ANALYTICS DASHBOARD PAGE
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Business Analytics Dashboard")
    st.markdown("Comprehensive insights into customer behavior, product performance, and business metrics.")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    # Customer segment distribution
    if not data['customer_segments'].empty and 'Segment' in data['customer_segments'].columns:
        st.markdown("### 👥 Customer Segment Distribution")
        
        segment_counts = data['customer_segments']['Segment'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution by Segment"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Segment Breakdown:**")
            total_customers = len(data['customer_segments'])
            for segment, count in segment_counts.items():
                percentage = (count / total_customers) * 100
                segment_class = get_segment_color_class(segment)
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <span class="segment-label {segment_class}" style="font-size: 0.8rem;">
                        {segment}
                    </span>
                    <br>{count:,} customers ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Demo segment distribution
        st.markdown("### 👥 Demo: Customer Segment Distribution")
        
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
                title="Demo: Customer Distribution by Segment"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Demo Segment Breakdown:**")
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
    
    # Product performance
    if not data['product_info'].empty:
        st.markdown("### 📦 Product Performance")
        
        top_products = data['product_info'].head(10)
        
        if 'Customer_Count' in top_products.columns:
            fig = px.bar(
                top_products,
                x='Customer_Count',
                y='Description',
                orientation='h',
                title="Top 10 Products by Customer Count"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product performance data is not available.")
    
    # Business insights
    st.markdown("### 💡 Key Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Customer Insights:**
        - Strong customer base with diverse segments
        - High-value customers drive significant revenue
        - Opportunity to convert at-risk customers
        - New customer acquisition is steady
        """)
    
    with col2:
        st.markdown("""
        **Product Insights:**
        - Home decoration items are popular
        - White/cream colored products perform well
        - Heart-themed products show strong demand
        - Seasonal items drive engagement
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🛒 <strong>Shopper Spectrum</strong> - E-Commerce Customer Analytics Platform</p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)