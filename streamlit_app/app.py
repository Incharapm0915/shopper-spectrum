import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
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

# Helper functions for loading compressed files
def load_compressed_pickle(file_path):
    """Load a compressed pickle file"""
    try:
        if file_path.endswith('.pkl.gz'):
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

def load_csv_with_fallback(primary_path, fallback_path=None):
    """Load CSV with fallback option"""
    try:
        return pd.read_csv(primary_path)
    except Exception as e:
        if fallback_path:
            try:
                return pd.read_csv(fallback_path)
            except Exception as e2:
                st.error(f"Error loading CSV files: {e}, {e2}")
                return pd.DataFrame()
        else:
            st.error(f"Error loading {primary_path}: {e}")
            return pd.DataFrame()

# Load models and data
@st.cache_data
def load_models_and_data():
    """Load all required models and data files (compressed versions)"""
    try:
        data = {}
        
        # Load compressed models
        model_files = {
            'product_recommendations': 'models/recommendations_reduced_compressed.pkl.gz',
            'product_name_mapping': 'models/product_name_mapping_compressed.pkl.gz',
            'popular_products': 'models/popular_products_compressed.pkl.gz',
            'kmeans_model': 'models/kmeans_model_compressed.pkl.gz',
            'scaler': 'models/scaler_compressed.pkl.gz',
            'cluster_mapping': 'models/cluster_mapping_compressed.pkl.gz',
            'rec_metadata': 'models/recommendation_metadata_compressed.pkl.gz'
        }
        
        # Load each model with fallback to uncompressed version
        for key, compressed_path in model_files.items():
            fallback_path = compressed_path.replace('_compressed.pkl.gz', '.pkl')
            data[key] = load_compressed_pickle(compressed_path)
            if data[key] is None:
                data[key] = load_compressed_pickle(fallback_path)
        
        # Load compressed CSV files
        csv_files = {
            'product_info': ('data/processed/product_info_compressed.csv', 'data/processed/product_info.csv'),
            'segment_insights': ('data/processed/segment_insights_compressed.csv', 'data/processed/segment_insights.csv'),
            'eda_summary': ('data/processed/eda_summary_compressed.csv', 'data/processed/eda_summary.csv'),
            'customer_segments': ('data/processed/customer_segments_compressed.csv', 'data/processed/customer_segments.csv')
        }
        
        for key, (primary_path, fallback_path) in csv_files.items():
            data[key] = load_csv_with_fallback(primary_path, fallback_path)
        
        # Create default metadata if not available
        if data['rec_metadata'] is None:
            data['rec_metadata'] = {
                'total_products': len(data['product_info']) if not data['product_info'].empty else 0,
                'recommendable_products': len(data['product_recommendations']) if data['product_recommendations'] else 0,
                'total_customers': len(data['customer_segments']) if not data['customer_segments'].empty else 0
            }
        
        # Validate essential components
        essential_items = ['kmeans_model', 'scaler', 'cluster_mapping']
        missing_items = [item for item in essential_items if data.get(item) is None]
        
        if missing_items:
            st.error(f"Failed to load essential components: {missing_items}")
            return None
        
        return data
        
    except Exception as e:
        st.error(f"Error loading models and data: {e}")
        return None

def get_segment_color_class(segment):
    """Get CSS class for segment label"""
    segment_lower = segment.lower().replace(' ', '-').replace('_', '-')
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
                product_row = product_info[product_info['StockCode'] == stock_code]
                if not product_row.empty:
                    rec_details.append({
                        'StockCode': stock_code,
                        'Description': product_row['Description'].iloc[0],
                        'Similarity_Score': similarity_score,
                        'Customer_Count': product_row.get('Customer_Count', [0]).iloc[0] if 'Customer_Count' in product_row.columns else 0,
                        'Total_Revenue': product_row.get('Total_Revenue', [0]).iloc[0] if 'Total_Revenue' in product_row.columns else 0
                    })
            return rec_details
        elif isinstance(recs, list):
            # Handle original recommendation format
            return recs[:n_recommendations]
        
        return []
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

# Main header
st.markdown('<h1 class="main-header">🛒 Shopper Spectrum</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Customer Segmentation and Product Recommendations in E-Commerce</p>', unsafe_allow_html=True)

# Load data
with st.spinner("Loading models and data..."):
    data = load_models_and_data()

if data is None:
    st.error("Failed to load required models. Please ensure all model files are in the 'models' directory.")
    st.stop()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "🎯 Product Recommender", "👥 Customer Segmentation", "📊 Analytics Dashboard"]
)

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 System Stats")
if isinstance(data['rec_metadata'], dict):
    st.sidebar.metric("Total Products", f"{data['rec_metadata'].get('total_products', 0):,}")
    st.sidebar.metric("Recommendable Products", f"{data['rec_metadata'].get('recommendable_products', 0):,}")
    st.sidebar.metric("Active Customers", f"{data['rec_metadata'].get('total_customers', 0):,}")

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
            value=eda_metrics.get('Total Customers', 'N/A')
        )
    
    with col2:
        st.metric(
            label="Total Revenue", 
            value=eda_metrics.get('Total Revenue', 'N/A')
        )
    
    with col3:
        st.metric(
            label="Average Customer Value", 
            value=eda_metrics.get('Average Customer Value', 'N/A')
        )
    
    with col4:
        st.metric(
            label="Total Products", 
            value=eda_metrics.get('Total Products', 'N/A')
        )
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Product Recommendations</h3>
            <p>Get intelligent product suggestions based on collaborative filtering and customer purchase patterns. Our system analyzes buying behavior to recommend similar products.</p>
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
            <p>Discover your customer segment using RFM (Recency, Frequency, Monetary) analysis and machine learning clustering algorithms.</p>
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
            <p>Explore comprehensive business insights with interactive visualizations and detailed analytics about customers, products, and trends.</p>
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
    
    if search_term or search_button:
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
                for i, result in enumerate(search_results[:5]):  # Show top 5 matches
                    with st.expander(f"📦 {result['Description']}", expanded=(i==0)):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Stock Code:** {result['StockCode']}")
                            st.write(f"**Customers who bought this:** {result['Customer_Count']}")
                        
                        with col2:
                            if st.button(f"Recommend Similar", key=f"rec_{result['StockCode']}"):
                                # Get recommendations for this product
                                recommendations = get_product_recommendations_safe(
                                    result['StockCode'], 
                                    data['product_recommendations'], 
                                    data['product_info']
                                )
                                
                                if recommendations:
                                    st.markdown(f"### 🎯 Recommendations for: {result['Description']}")
                                    
                                    # Display recommendations in cards
                                    for j, rec in enumerate(recommendations):
                                        st.markdown(f"""
                                        <div class="recommendation-card">
                                            <h4>#{j+1} {rec.get('Description', 'N/A')}</h4>
                                            <p><strong>Stock Code:</strong> {rec.get('StockCode', 'N/A')}</p>
                                            <p><strong>Similarity Score:</strong> {rec.get('Similarity_Score', 0):.3f}</p>
                                            <p><strong>Purchased by:</strong> {rec.get('Customer_Count', 0)} customers</p>
                                            <p><strong>Total Revenue:</strong> ${rec.get('Total_Revenue', 0):,.2f}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.warning("No recommendations available for this product.")
            else:
                st.warning("No products found matching your search. Try different keywords.")
                
                # Show popular products as fallback
                if data['popular_products']:
                    st.markdown("### 🌟 Popular Products You Might Like")
                    for product in data['popular_products'][:5]:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{product.get('Description', 'N/A')}</h4>
                            <p><strong>Stock Code:</strong> {product.get('StockCode', 'N/A')}</p>
                            <p><strong>Purchased by:</strong> {product.get('Customer_Count', 0)} customers</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Show example searches
    st.markdown("---")
    st.markdown("### 💡 Try These Example Searches:")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("🤍 WHITE ITEMS"):
            st.rerun()
    
    with example_col2:
        if st.button("🎄 CHRISTMAS"):
            st.rerun()
    
    with example_col3:
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
            # Predict customer segment
            try:
                # Prepare the data
                rfm_data = np.array([[recency, frequency, monetary]])
                
                # Scale the data
                rfm_scaled = data['scaler'].transform(rfm_data)
                
                # Predict cluster
                cluster = data['kmeans_model'].predict(rfm_scaled)[0]
                
                # Get segment label
                segment = data['cluster_mapping']['cluster_to_segment'][cluster]
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎉 Your Customer Segment")
                
                # Create segment label with styling
                segment_class = get_segment_color_class(segment)
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <span class="segment-label {segment_class}" style="font-size: 1.5rem;">
                        {segment}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Show segment insights if available
                if not data['segment_insights'].empty:
                    segment_data = data['segment_insights'][data['segment_insights'].index == segment]
                    
                    if not segment_data.empty:
                        st.markdown("### 📈 Segment Characteristics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Average Recency", 
                                f"{segment_data['Avg_Recency'].iloc[0]:.0f} days"
                            )
                        
                        with col2:
                            st.metric(
                                "Average Frequency", 
                                f"{segment_data['Avg_Frequency'].iloc[0]:.1f} orders"
                            )
                        
                        with col3:
                            st.metric(
                                "Average Spend", 
                                f"${segment_data['Avg_Monetary'].iloc[0]:,.2f}"
                            )
                        
                        with col4:
                            st.metric(
                                "Customer Count", 
                                f"{segment_data['Customer_Count'].iloc[0]:,}"
                            )
                
                # Segment-specific recommendations
                st.markdown("### 💡 Personalized Recommendations")
                
                recommendations_map = {
                    'Champions': [
                        "🏆 You're our most valuable customer! Consider our VIP membership program.",
                        "🎁 Get early access to new product launches and exclusive deals.",
                        "💎 Explore our premium product collection curated just for you.",
                        "📧 Help us improve by sharing your feedback and reviews."
                    ],
                    'Loyal Customers': [
                        "🤝 Thank you for your loyalty! Check out our customer rewards program.",
                        "📦 Try our subscription service for regular deliveries.",
                        "🆙 Explore higher-value products that match your interests.",
                        "👥 Refer friends and earn rewards for each successful referral."
                    ],
                    'At Risk': [
                        "💔 We miss you! Here's a special 20% discount to welcome you back.",
                        "📞 Let us know if there's anything we can improve about your experience.",
                        "🎯 Check out new products that match your previous purchases.",
                        "📧 Subscribe to our newsletter for exclusive deals and updates."
                    ],
                    'New Customers': [
                        "👋 Welcome! Here's a beginner's guide to our best products.",
                        "🎁 Enjoy 15% off your next purchase with code WELCOME15.",
                        "📱 Download our mobile app for exclusive app-only deals.",
                        "💬 Join our community forum to connect with other customers."
                    ],
                    'Regular Customers': [
                        "⭐ Thanks for being a regular customer! Try our monthly deals.",
                        "🔄 Set up automatic reorders for your frequently bought items.",
                        "📊 View your purchase history to discover new favorites.",
                        "🎯 Get personalized product recommendations based on your preferences."
                    ],
                    'Price Sensitive': [
                        "💰 Check out our current sales and clearance items.",
                        "📈 Consider bulk purchases for better value on your favorites.",
                        "🏷️ Sign up for price drop alerts on products you're watching.",
                        "💳 Learn about our payment plans for larger purchases."
                    ]
                }
                
                recs = recommendations_map.get(segment, ["Thank you for being our customer!"])
                
                for i, rec in enumerate(recs, 1):
                    st.markdown(f"{i}. {rec}")
                
            except Exception as e:
                st.error(f"Error predicting segment: {e}")
    
    # Segment overview
    st.markdown("---")
    st.markdown("### 🎯 Customer Segment Overview")
    
    # Display all segments with their characteristics
    if not data['segment_insights'].empty:
        for segment in data['segment_insights'].index:
            with st.expander(f"📊 {segment} Segment"):
                segment_data = data['segment_insights'].loc[segment]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Average Metrics:**
                    - Recency: {segment_data.get('Avg_Recency', 0):.0f} days
                    - Frequency: {segment_data.get('Avg_Frequency', 0):.1f} orders
                    - Monetary: ${segment_data.get('Avg_Monetary', 0):,.2f}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Segment Size:**
                    - Customers: {segment_data.get('Customer_Count', 0):,}
                    - Revenue Share: {segment_data.get('Revenue_Share_Pct', 0):.1f}%
                    - Customer Share: {segment_data.get('Customer_Share_Pct', 0):.1f}%
                    """)

# ANALYTICS DASHBOARD PAGE
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Business Analytics Dashboard")
    st.markdown("Comprehensive insights into customer behavior, product performance, and business metrics.")
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    # Load additional data for analytics
    try:
        customer_segments = data['customer_segments']
        
        if not customer_segments.empty:
            with col1:
                total_customers = len(customer_segments)
                st.metric("Total Customers", f"{total_customers:,}")
            
            with col2:
                if 'Monetary' in customer_segments.columns:
                    total_revenue = customer_segments['Monetary'].sum()
                    st.metric("Total Revenue", f"${total_revenue:,.2f}")
                else:
                    st.metric("Total Revenue", "N/A")
            
            with col3:
                if 'Monetary' in customer_segments.columns:
                    avg_customer_value = customer_segments['Monetary'].mean()
                    st.metric("Avg Customer Value", f"${avg_customer_value:.2f}")
                else:
                    st.metric("Avg Customer Value", "N/A")
            
            with col4:
                if 'Segment' in customer_segments.columns:
                    segments_count = customer_segments['Segment'].nunique()
                    st.metric("Customer Segments", segments_count)
                else:
                    st.metric("Customer Segments", "N/A")
            
            # Segment distribution chart
            if 'Segment' in customer_segments.columns:
                st.markdown("### 👥 Customer Segment Distribution")
                
                segment_counts = customer_segments['Segment'].value_counts()
                
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
            
            # RFM distribution analysis
            rfm_columns = ['Recency', 'Frequency', 'Monetary']
            available_rfm = [col for col in rfm_columns if col in customer_segments.columns]
            
            if available_rfm:
                st.markdown("### 📊 RFM Analysis")
                
                cols = st.columns(len(available_rfm))
                
                for i, col in enumerate(available_rfm):
                    with cols[i]:
                        fig = px.histogram(
                            customer_segments, 
                            x=col, 
                            nbins=30,
                            title=f"{col} Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Product performance
        if not data['product_info'].empty:
            st.markdown("### 📦 Product Performance")
            
            top_products = data['product_info'].head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
                if 'Total_Revenue' in top_products.columns:
                    fig = px.bar(
                        top_products,
                        x='Total_Revenue',
                        y='Description',
                        orientation='h',
                        title="Top 10 Products by Revenue"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Revenue data not available for products")
        
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        st.info("Some analytics features may not be available with compressed data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🛒 <strong>Shopper Spectrum</strong> - E-Commerce Customer Analytics Platform</p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)