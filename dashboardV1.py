import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from datetime import datetime
import re
from collections import Counter
from textblob import TextBlob
from keybert import KeyBERT
import warnings
import ast
import folium
from streamlit_folium import st_folium
warnings.filterwarnings('ignore')

# Initialize KeyBERT model
@st.cache_resource
def load_keybert_model():
    """Load and cache KeyBERT model"""
    try:
        # Use a lightweight multilingual model that works well for Indonesian
        kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
        return kw_model
    except Exception as e:
        st.error(f"❌ Error loading KeyBERT model: {str(e)}")
        st.info("💡 Tip: Pastikan koneksi internet stabil untuk download model pertama kali.")
        return None

# Helper function to safely parse list columns
def safe_parse_list(value):
    """Safely parse string representation of lists"""
    if pd.isna(value) or value == '' or value == '[]':
        return []
    try:
        if isinstance(value, str):
            # Remove quotes and brackets, then split
            value = value.strip('[]').replace("'", "").replace('"', '')
            if value:
                return [item.strip() for item in value.split(',') if item.strip()]
        return []
    except:
        return []

# Helper function to ensure numeric columns have valid values
def ensure_numeric_columns(df, columns, default_value=0):
    """
    Ensures that specified columns exist and have valid numeric values.
    
    Args:
        df: DataFrame to process
        columns: List of columns to ensure
        default_value: Default value for missing columns
    
    Returns:
        DataFrame with ensured columns
    """
    result = df.copy()
    
    for col in columns:
        # If column doesn't exist, create it with default value
        if col not in result.columns:
            result[col] = default_value
        else:
            # Ensure column is numeric
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(default_value)
            
            # Replace zeros with small value to avoid division by zero
            if col in ['views_count'] and (result[col] == 0).any():
                result[col] = result[col].replace(0, 1)
                
    return result

# Page configuration
st.set_page_config(
    page_title="Dashboard Analisis Media Sosial V1.0",
    page_icon="📊",
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
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
    .location-info {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }


</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        # Load sentiment data
        sentiment_df = pd.read_csv('combined_social_media_data_fixed.csv')
        
        # Load SNA data
        sna_df = pd.read_csv('combined_sna_relations_fixed.csv')
        
        # Convert timestamp columns with improved handling for various formats
        try:
            # Try ISO8601 format first (handles Z suffix)
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], format='ISO8601')
        except:
            try:
                # Fallback to mixed format parsing
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], format='mixed')
            except:
                # Last resort: infer format
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], infer_datetime_format=True)
        
        try:
            # Try ISO8601 format first for SNA data
            sna_df['timestamp'] = pd.to_datetime(sna_df['timestamp'], format='ISO8601')
        except:
            try:
                # Fallback to mixed format parsing
                sna_df['timestamp'] = pd.to_datetime(sna_df['timestamp'], format='mixed')
            except:
                # Last resort: infer format
                sna_df['timestamp'] = pd.to_datetime(sna_df['timestamp'], infer_datetime_format=True)
        
        # Parse list columns in sentiment data
        list_columns = ['hashtags', 'mentions']
        for col in list_columns:
            if col in sentiment_df.columns:
                sentiment_df[col + '_parsed'] = sentiment_df[col].apply(safe_parse_list)
        
        # Parse NER columns
        ner_columns = ['PER', 'ORG', 'PET', 'SPORT_HOBBY', 'INFLUENCER']
        for col in ner_columns:
            if col in sentiment_df.columns:
                sentiment_df[col + '_parsed'] = sentiment_df[col].apply(safe_parse_list)
        
        # Filter out empty content_text_cleaned data
        initial_sentiment_count = len(sentiment_df)
        initial_sna_count = len(sna_df)
        
        sentiment_df = sentiment_df[sentiment_df['content_text_cleaned'].notna() & 
                                  (sentiment_df['content_text_cleaned'].str.strip() != '')]
        sna_df = sna_df[sna_df['content_text_cleaned'].notna() & 
                       (sna_df['content_text_cleaned'].str.strip() != '')]
        
        cleaned_sentiment_count = len(sentiment_df)
        cleaned_sna_count = len(sna_df)
        
        st.success(f"✅ Data berhasil dimuat dan dibersihkan:")
        st.info(f"📊 Sentiment: {cleaned_sentiment_count:,} posts (filtered {initial_sentiment_count - cleaned_sentiment_count:,} empty content)")
        st.info(f"🕸️ SNA: {cleaned_sna_count:,} relations (filtered {initial_sna_count - cleaned_sna_count:,} empty content)")
        
        return sentiment_df, sna_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("💡 Tip: Pastikan file CSV tersedia dan format timestamp sesuai")
        return None, None

def extract_topics(text_series, n_topics=10):
    """Extract top topics from text data using KeyBERT"""
    if text_series.empty:
        return []
    
    # Load KeyBERT model
    kw_model = load_keybert_model()
    if kw_model is None:
        # Fallback to frequency-based approach if KeyBERT fails
        return extract_topics_fallback(text_series, n_topics)
    
    try:
        # Combine all text
        text = ' '.join(text_series.fillna('').astype(str))
        
        # Clean text: Remove mentions, hashtags, URLs
        text = re.sub(r'@\w+|#\w+|http\S+|www\S+|RT\s+', '', text)
        
        # Remove extra whitespace and non-alphabetic characters except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 50:  # If text too short, fallback
            return extract_topics_fallback(text_series, n_topics)
        
        # Extract keywords/keyphrases using KeyBERT
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=20,
            diversity=0.5
        )[:n_topics]
        
        # Convert to format (topic, score)
        topics = [(keyword, round(score * 100)) for keyword, score in keywords]
        
        # Filter out very short keywords and common words
        indonesian_stopwords = {'yang', 'dan', 'untuk', 'dari', 'dengan', 'pada', 'ini', 'itu', 'adalah', 'akan', 'sudah', 'tidak', 'ada', 'juga', 'bisa', 'saya', 'kita', 'mereka', 'dalam', 'atau', 'the', 'and', 'or', 'to', 'of', 'in', 'for', 'is', 'are', 'was', 'were', 'have', 'has', 'can', 'will', 'would', 'could', 'should'}
        
        filtered_topics = []
        for topic, score in topics:
            topic_words = topic.lower().split()
            if (len(topic.strip()) >= 3 and 
                not any(word in indonesian_stopwords for word in topic_words) and
                not topic.lower().isdigit()):
                filtered_topics.append((topic, score))
        
        return filtered_topics[:n_topics] if filtered_topics else extract_topics_fallback(text_series, n_topics)
    
    except Exception as e:
        st.warning(f"⚠️ KeyBERT extraction failed: {str(e)}. Menggunakan metode fallback.")
        return extract_topics_fallback(text_series, n_topics)

def extract_topics_fallback(text_series, n_topics=10):
    """Fallback topic extraction using word frequency"""
    if text_series.empty:
        return []
    
    # Combine all text
    text = ' '.join(text_series.fillna('').astype(str))
    
    # Simple topic extraction using word frequency
    text = re.sub(r'@\w+|#\w+|http\S+|www\S+', '', text)
    
    # Split into words and count frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter common words
    stop_words = {'dan', 'atau', 'yang', 'untuk', 'dari', 'ini', 'itu', 'dengan', 'pada', 'di', 'ke', 'dalam', 'adalah', 'akan', 'sudah', 'tidak', 'ada', 'juga', 'bisa', 'saya', 'kita', 'mereka', 'the', 'and', 'or', 'to', 'of', 'in', 'for', 'is', 'are', 'was', 'were'}
    words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Get top words
    word_counts = Counter(words)
    return word_counts.most_common(n_topics)

def create_wordcloud(text_series, title="WordCloud"):
    """Create wordcloud from text data"""
    if text_series.empty:
        return None
    
    text = ' '.join(text_series.fillna('').astype(str))
    
    # Remove mentions, hashtags, URLs
    text = re.sub(r'@\w+|#\w+|http\S+|www\S+', '', text)
    
    if not text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    return fig

def create_network_graph(sna_df):
    """Create network graph for SNA analysis"""
    if sna_df.empty:
        return None, [], None
    
    # Create network graph with weighted edges
    G = nx.DiGraph()
    
    # Count relationships between nodes
    edge_weights = {}
    
    for _, row in sna_df.iterrows():
        source = row['source']
        target = row['target']
        relation = row['relation']
        
        if pd.notna(source) and pd.notna(target):
            edge_key = (source, target)
            if edge_key in edge_weights:
                edge_weights[edge_key] += 1
            else:
                edge_weights[edge_key] = 1
            
            # Add edge with weight and relation
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
                G[source][target]['relations'].append(relation)
            else:
                G.add_edge(source, target, weight=1, relations=[relation])
    
    if G.number_of_nodes() == 0:
        return None, [], None
    
    # Calculate influence metrics
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    weighted_in_degree = dict(G.in_degree(weight='weight'))
    
    # Top influencers
    top_influencers = sorted(weighted_in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Create network visualization
    network_fig = create_network_visualization(G)
    
    return G, top_influencers, network_fig

def create_network_visualization(G):
    """Create network visualization using matplotlib"""
    if G.number_of_nodes() == 0:
        return None
    
    # Limit nodes for better visualization
    if G.number_of_nodes() > 50:
        weighted_degrees = dict(G.degree(weight='weight'))
        top_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:50]
        top_node_names = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_names)
    
    # Create layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=100, weight='weight')
    except:
        pos = nx.random_layout(G)
    
    # Calculate node sizes and colors
    weighted_in_degrees = dict(G.in_degree(weight='weight'))
    max_weighted_in_degree = max(weighted_in_degrees.values()) if weighted_in_degrees.values() else 1
    min_weighted_in_degree = min(weighted_in_degrees.values()) if weighted_in_degrees.values() else 0
    
    node_sizes = []
    for node in G.nodes():
        if max_weighted_in_degree > min_weighted_in_degree:
            normalized = (weighted_in_degrees[node] - min_weighted_in_degree) / (max_weighted_in_degree - min_weighted_in_degree)
            size = 200 + (normalized * 1800)
        else:
            size = 600
        node_sizes.append(size)
    
    node_colors = [weighted_in_degrees[node] / max_weighted_in_degree for node in G.nodes()]
    
    # Calculate edge widths
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    min_edge_weight = min(edge_weights) if edge_weights else 1
    
    edge_widths = []
    for weight in edge_weights:
        if max_edge_weight > min_edge_weight:
            normalized = (weight - min_edge_weight) / (max_edge_weight - min_edge_weight)
            width = 0.5 + (normalized * 4.5)
        else:
            width = 2
        edge_widths.append(width)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          alpha=0.6,
                          arrows=True,
                          arrowsize=15,
                          arrowstyle='-|>',
                          width=edge_widths,
                          ax=ax)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors,
                          cmap=plt.cm.plasma,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1,
                          ax=ax)
    
    # Add labels for top nodes
    weighted_degrees = dict(G.degree(weight='weight'))
    top_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
    top_node_names = [node for node, _ in top_nodes]
    labels = {node: node[:15] + '...' if len(node) > 15 else node for node in top_node_names}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title("Network Visualization\n(Node size = Influence, Edge thickness = Relationship strength)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                              norm=plt.Normalize(vmin=min_weighted_in_degree, vmax=max_weighted_in_degree))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Weighted In-Degree (Influence Level)', rotation=270, labelpad=20)
    
    # Add legend
    legend_text = f"Edge Thickness: Min={min_edge_weight}, Max={max_edge_weight} connections"
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig



def group_sna_relations(relation, platform=None):
    """Group SNA relations for simplified analysis"""
    if pd.isna(relation):
        return "Other"
    
    relation_str = str(relation).lower()
    platform_str = str(platform).lower() if platform else ""
    
    # Group ALL mentions together (including the ones that were going to "Other")
    if 'mention' in relation_str:
        return "Mentions (NER)"
    
    # Group hashtag usage
    elif 'hashtag' in relation_str or 'uses_hashtag' in relation_str:
        return "Hashtag Usage"
    
    # For Twitter: Replies + Retweets = Sharing
    elif platform_str == 'twitter' and ('reply' in relation_str or 'retweet' in relation_str):
        return "Sharing"
    
    # For other platforms: Replies/Comments = Interactions
    elif 'reply' in relation_str or 'comment' in relation_str:
        return "Interactions"
    
    # Sharing activities
    elif 'share' in relation_str or 'retweet' in relation_str:
        return "Sharing"
    
    else:
        return "Other"


def create_geo_map(df, location_col='primary_location'):
    """Create a geographical map showing post distribution"""
    # Filter data with valid locations
    geo_data = df[df[location_col].notna() & (df[location_col] != '')]
    
    if geo_data.empty:
        return None
    
    # Count posts by location
    location_counts = geo_data[location_col].value_counts()
    
    # For simplicity, use a basic map centered on Indonesia
    # In a real implementation, you'd want to geocode the locations
    m = folium.Map(location=[-2.5, 118], zoom_start=5)
    
    # Add markers for top locations (this is a simplified version)
    # In practice, you'd want to geocode the actual locations
    for location, count in location_counts.head(10).items():
        # This is a placeholder - you'd need actual coordinates
        folium.CircleMarker(
            location=[-2.5 + np.random.uniform(-5, 5), 118 + np.random.uniform(-10, 10)],
            radius=min(count / 10, 50),
            popup=f"{location}: {count} posts",
            color='red',
            fill=True,
            fillColor='red'
        ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Dashboard Analisis Media Sosial V1.0</h1>', unsafe_allow_html=True)
    st.markdown("### ✨ Enhanced dengan Analisis Geografis dan Named Entity Recognition")
    
    # Load data
    sentiment_df, sna_df = load_data()
    
    if sentiment_df is None or sna_df is None:
        st.error("Gagal memuat data. Pastikan file CSV tersedia.")
        return
    
    # Sidebar for filters
    st.sidebar.header("🔧 Filter & Pengaturan")
    
    # Enhanced filters
    
    # Platform filter
    all_platforms = sentiment_df['platform'].unique().tolist() if 'platform' in sentiment_df.columns else []
    selected_platforms = st.sidebar.multiselect(
        "Pilih Platform",
        options=all_platforms,
        default=all_platforms[:3] if len(all_platforms) >= 3 else all_platforms
    )
    
    # Province filter (NEW)
    all_provinces = sentiment_df['province'].unique().tolist() if 'province' in sentiment_df.columns else []
    all_provinces = [p for p in all_provinces if pd.notna(p) and p != 'unknown']
    selected_provinces = st.sidebar.multiselect(
        "🗺️ Pilih Provinsi",
        options=['Semua'] + all_provinces,
        default=['Semua']
    )
    
    # Confidence threshold filter (NEW) - DISABLED per user request
    # st.sidebar.subheader("🎯 Confidence Threshold")
    sentiment_confidence = 0.0  # Default to include all data
    emotion_confidence = 0.0    # Default to include all data
    
    # Keyword filter (NEW)
    keyword_filter = st.sidebar.text_input("🔍 Filter berdasarkan Keyword")
    
    # Date range filter
    if not sentiment_df.empty and 'timestamp' in sentiment_df.columns:
        min_date = sentiment_df['timestamp'].min().date()
        max_date = sentiment_df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Pilih Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Apply filters
    filtered_sentiment = sentiment_df.copy()
    filtered_sna = sna_df.copy()
    
    if selected_platforms:
        filtered_sentiment = filtered_sentiment[filtered_sentiment['platform'].isin(selected_platforms)]
        filtered_sna = filtered_sna[filtered_sna['platform'].isin(selected_platforms)]
    
    if 'Semua' not in selected_provinces and selected_provinces:
        filtered_sentiment = filtered_sentiment[filtered_sentiment['province'].isin(selected_provinces)]
        filtered_sna = filtered_sna[filtered_sna['province'].isin(selected_provinces)]
    
    # Confidence filters disabled per user request
    # All data will be included without confidence filtering
    
    # Apply keyword filter
    if keyword_filter:
        filtered_sentiment = filtered_sentiment[
            filtered_sentiment['content_text_cleaned'].str.contains(keyword_filter, case=False, na=False)
        ]
    
    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_sentiment = filtered_sentiment[
            (filtered_sentiment['timestamp'].dt.date >= start_date) & 
            (filtered_sentiment['timestamp'].dt.date <= end_date)
        ]
        filtered_sna = filtered_sna[
            (filtered_sna['timestamp'].dt.date >= start_date) & 
            (filtered_sna['timestamp'].dt.date <= end_date)
        ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Overview", "🗺️ Geografis", "🤖 NER Analysis", "🎯 Engagement", 
        "😊 Sentimen", "🎭 Emotion", "☁️ WordCloud", "🕸️ SNA", "🔥 Trending"
    ])
    
    with tab1:
        st.header("📈 Overview Dashboard")
        
        # Enhanced key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_posts = len(filtered_sentiment)
            st.metric("Total Posts", f"{total_posts:,}")
        
        with col2:
            if 'views_count' in filtered_sentiment.columns:
                total_views = filtered_sentiment['views_count'].sum()
                st.metric("Total Views", f"{total_views:,}")
        
        with col3:
            if 'likes_count' in filtered_sentiment.columns:
                total_likes = filtered_sentiment['likes_count'].sum()
                st.metric("Total Likes", f"{total_likes:,}")
        
        with col4:
            platforms_count = len(filtered_sentiment['platform'].unique())
            st.metric("Platforms", platforms_count)
        
        with col5:
            if 'province' in filtered_sentiment.columns:
                provinces_count = len(filtered_sentiment[filtered_sentiment['province'] != 'unknown']['province'].unique())
                st.metric("Provinsi", provinces_count)
        
        # Platform and Province distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if not filtered_sentiment.empty:
                st.subheader("Distribusi Platform")
                platform_counts = filtered_sentiment['platform'].value_counts()
                
                fig = px.pie(
                    values=platform_counts.values,
                    names=platform_counts.index,
                    title="Distribusi Posts per Platform"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'province' in filtered_sentiment.columns:
                st.subheader("Distribusi Provinsi")
                province_counts = filtered_sentiment[filtered_sentiment['province'] != 'unknown']['province'].value_counts().head(10)
                
                if not province_counts.empty:
                    fig = px.bar(
                        x=province_counts.values,
                        y=province_counts.index,
                        orientation='h',
                        title="Top 10 Provinsi"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Video analytics (TikTok specific)
        if 'video_duration' in filtered_sentiment.columns:
            st.subheader("📹 Analisis Video (TikTok)")
            
            video_data = filtered_sentiment[filtered_sentiment['platform'] == 'tiktok'].copy()
            if not video_data.empty and 'video_duration' in video_data.columns:
                video_data['video_duration'] = pd.to_numeric(video_data['video_duration'], errors='coerce')
                video_data = video_data.dropna(subset=['video_duration'])
                
                if not video_data.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_duration = video_data['video_duration'].mean()
                        st.metric("Rata-rata Durasi Video", f"{avg_duration:.1f}s")
                    
                    with col2:
                        total_videos = len(video_data)
                        st.metric("Total Video", f"{total_videos:,}")
                    
                    with col3:
                        if 'views_count' in video_data.columns:
                            avg_views_per_second = (video_data['views_count'] / video_data['video_duration']).mean()
                            st.metric("Avg Views/Detik", f"{avg_views_per_second:.0f}")
                    
                    # Duration vs Engagement
                    if 'views_count' in video_data.columns:
                        fig = px.scatter(
                            video_data,
                            x='video_duration',
                            y='views_count',
                            title="Durasi Video vs Views"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🗺️ Analisis Geografis")
        
        if 'primary_location' in filtered_sentiment.columns or 'province' in filtered_sentiment.columns:
            # Location overview
            st.subheader("📍 Overview Lokasi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'primary_location' in filtered_sentiment.columns:
                    unique_locations = filtered_sentiment['primary_location'].nunique()
                    st.metric("Unique Primary Locations", unique_locations)
            
            with col2:
                if 'secondary_location' in filtered_sentiment.columns:
                    unique_secondary = filtered_sentiment['secondary_location'].nunique()
                    st.metric("Unique Secondary Locations", unique_secondary)
            
            with col3:
                if 'province' in filtered_sentiment.columns:
                    known_provinces = len(filtered_sentiment[filtered_sentiment['province'] != 'unknown']['province'].unique())
                    st.metric("Provinsi Teridentifikasi", known_provinces)
            
            # Province analysis
            if 'province' in filtered_sentiment.columns:
                st.subheader("📊 Analisis per Provinsi")
                
                province_data = filtered_sentiment[filtered_sentiment['province'] != 'unknown'].copy()
                if not province_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Posts by province
                        province_counts = province_data['province'].value_counts().head(15)
                        fig = px.bar(
                            x=province_counts.values,
                            y=province_counts.index,
                            orientation='h',
                            title="Jumlah Posts per Provinsi",
                            color=province_counts.values,
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Sentiment by province
                        if 'sentiment' in province_data.columns:
                            sentiment_province = province_data.groupby(['province', 'sentiment']).size().reset_index(name='count')
                            top_provinces = province_counts.head(10).index.tolist()
                            sentiment_province_filtered = sentiment_province[sentiment_province['province'].isin(top_provinces)]
                            
                            fig = px.bar(
                                sentiment_province_filtered,
                                x='province',
                                y='count',
                                color='sentiment',
                                title="Sentimen per Provinsi (Top 10)",
                                barmode='stack'
                            )
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed province analysis
                    st.subheader("🔍 Analisis Detail per Provinsi")
                    
                    selected_province = st.selectbox(
                        "Pilih Provinsi untuk Analisis Detail",
                        options=province_counts.index.tolist(),
                        key="province_detail"
                    )
                    
                    if selected_province:
                        province_detail = province_data[province_data['province'] == selected_province]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Posts", len(province_detail))
                        
                        with col2:
                            if 'views_count' in province_detail.columns:
                                total_views = province_detail['views_count'].sum()
                                st.metric("Total Views", f"{total_views:,}")
                        
                        with col3:
                            if 'sentiment' in province_detail.columns:
                                dominant_sentiment = province_detail['sentiment'].mode().iloc[0]
                                st.metric("Sentimen Dominan", dominant_sentiment)
                        
                        with col4:
                            platforms_in_province = province_detail['platform'].nunique()
                            st.metric("Platform Aktif", platforms_in_province)
                        
                        # Trending topics in province
                        if 'content_text_cleaned' in province_detail.columns:
                            st.write(f"**🔥 Trending Topics di {selected_province}:**")
                            province_topics = extract_topics(province_detail['content_text_cleaned'], n_topics=10)
                            
                            if province_topics:
                                topics_df = pd.DataFrame(province_topics, columns=['Topic', 'Score'])
                                st.dataframe(topics_df, use_container_width=True)
            
            # Location-based insights
            if 'primary_location' in filtered_sentiment.columns:
                st.subheader("📍 Lokasi Spesifik")
                
                location_data = filtered_sentiment[filtered_sentiment['primary_location'].notna()]
                if not location_data.empty:
                    location_counts = location_data['primary_location'].value_counts().head(20)
                    
                    fig = px.bar(
                        x=location_counts.values,
                        y=location_counts.index,
                        orientation='h',
                        title="Top 20 Lokasi Spesifik",
                        color=location_counts.values,
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Data lokasi tidak tersedia dalam dataset ini")
    
    with tab3:
        st.header("🤖 Named Entity Recognition Analysis")
        
        # Check if NER columns exist
        ner_columns = ['PER_parsed', 'ORG_parsed', 'PET_parsed', 'SPORT_HOBBY_parsed', 'INFLUENCER_parsed']
        available_ner = [col for col in ner_columns if col in filtered_sentiment.columns]
        
        if available_ner:
            st.subheader("📊 Overview NER Entities")
            
            # NER Statistics
            ner_stats = {}
            for col in available_ner:
                # Count total entities
                total_entities = 0
                unique_entities = set()
                
                for entities in filtered_sentiment[col].dropna():
                    if isinstance(entities, list):
                        total_entities += len(entities)
                        unique_entities.update(entities)
                
                ner_stats[col.replace('_parsed', '')] = {
                    'total': total_entities,
                    'unique': len(unique_entities)
                }
            
            # Display statistics
            cols = st.columns(len(ner_stats))
            for i, (entity_type, stats) in enumerate(ner_stats.items()):
                with cols[i]:
                    st.metric(
                        f"{entity_type} Entities",
                        f"{stats['unique']} unique",
                        f"{stats['total']} total"
                    )
            
            # Top entities by type
            st.subheader("🏆 Top Entities per Kategori")
            
            for col in available_ner:
                entity_type = col.replace('_parsed', '')
                st.write(f"**{entity_type}:**")
                
                # Flatten all entities of this type
                all_entities = []
                for entities in filtered_sentiment[col].dropna():
                    if isinstance(entities, list):
                        all_entities.extend(entities)
                
                if all_entities:
                    entity_counts = Counter(all_entities)
                    top_entities = entity_counts.most_common(10)
                    
                    # Create visualization
                    if top_entities:
                        entities, counts = zip(*top_entities)
                        fig = px.bar(
                            x=list(counts),
                            y=list(entities),
                            orientation='h',
                            title=f"Top 10 {entity_type} Entities",
                            color=list(counts),
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(
                            yaxis={'categoryorder':'total ascending'},
                            showlegend=False,
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Tidak ada entities {entity_type} yang ditemukan")
            
            # Entity co-occurrence analysis
            st.subheader("🔗 Analisis Ko-Kejadian Entity")
            
            if 'PER_parsed' in filtered_sentiment.columns and 'ORG_parsed' in filtered_sentiment.columns:
                st.write("**Person-Organization Co-occurrence:**")
                
                co_occurrences = []
                for _, row in filtered_sentiment.iterrows():
                    persons = row.get('PER_parsed', []) or []
                    orgs = row.get('ORG_parsed', []) or []
                    
                    if isinstance(persons, list) and isinstance(orgs, list):
                        for person in persons:
                            for org in orgs:
                                co_occurrences.append((person, org))
                
                if co_occurrences:
                    co_occurrence_counts = Counter(co_occurrences)
                    top_co_occurrences = co_occurrence_counts.most_common(10)
                    
                    for (person, org), count in top_co_occurrences:
                        st.write(f"• {person} ↔ {org}: {count} kali")
            
            # Sentiment analysis by entity type
            if 'sentiment' in filtered_sentiment.columns:
                st.subheader("💭 Sentimen berdasarkan Entity Type")
                
                for col in available_ner[:3]:  # Limit to top 3 for readability
                    entity_type = col.replace('_parsed', '')
                    
                    # Get sentiment distribution for posts containing each entity type
                    has_entity = filtered_sentiment[col].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
                    entity_sentiment = filtered_sentiment[has_entity]['sentiment'].value_counts()
                    no_entity_sentiment = filtered_sentiment[~has_entity]['sentiment'].value_counts()
                    
                    if not entity_sentiment.empty and not no_entity_sentiment.empty:
                        comparison_data = pd.DataFrame({
                            f'With {entity_type}': entity_sentiment,
                            f'Without {entity_type}': no_entity_sentiment
                        }).fillna(0)
                        
                        # Normalize to percentages
                        comparison_data = comparison_data.div(comparison_data.sum()) * 100
                        
                        fig = px.bar(
                            comparison_data,
                            title=f"Distribusi Sentimen: Dengan vs Tanpa {entity_type} Entities (%)",
                            barmode='group'
                        )
                        fig.update_layout(showlegend=True)  # Keep legend for comparison chart
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Data Named Entity Recognition tidak tersedia dalam dataset ini")
    
    with tab4:
        st.header("🎯 Analisis Engagement")
        
        if not filtered_sentiment.empty:
            # Preprocessing: Combine replies + retweets into shares for Twitter
            filtered_sentiment_processed = filtered_sentiment.copy()
            
            # For Twitter: Combine replies_count + retweets_count into shares_count
            if 'platform' in filtered_sentiment_processed.columns:
                twitter_mask = filtered_sentiment_processed['platform'].str.lower() == 'twitter'
                
                if 'replies_count' in filtered_sentiment_processed.columns and 'retweets_count' in filtered_sentiment_processed.columns:
                    # For Twitter: Add replies + retweets to shares
                    if 'shares_count' not in filtered_sentiment_processed.columns:
                        filtered_sentiment_processed['shares_count'] = 0
                    
                    filtered_sentiment_processed.loc[twitter_mask, 'shares_count'] = (
                        filtered_sentiment_processed.loc[twitter_mask, 'shares_count'].fillna(0) +
                        filtered_sentiment_processed.loc[twitter_mask, 'replies_count'].fillna(0) +
                        filtered_sentiment_processed.loc[twitter_mask, 'retweets_count'].fillna(0)
                    )
                
                elif 'replies_count' in filtered_sentiment_processed.columns:
                    # If only replies_count exists, add to shares for Twitter
                    if 'shares_count' not in filtered_sentiment_processed.columns:
                        filtered_sentiment_processed['shares_count'] = 0
                    
                    filtered_sentiment_processed.loc[twitter_mask, 'shares_count'] = (
                        filtered_sentiment_processed.loc[twitter_mask, 'shares_count'].fillna(0) +
                        filtered_sentiment_processed.loc[twitter_mask, 'replies_count'].fillna(0)
                    )
                
                elif 'retweets_count' in filtered_sentiment_processed.columns:
                    # If only retweets_count exists, add to shares for Twitter
                    if 'shares_count' not in filtered_sentiment_processed.columns:
                        filtered_sentiment_processed['shares_count'] = 0
                    
                    filtered_sentiment_processed.loc[twitter_mask, 'shares_count'] = (
                        filtered_sentiment_processed.loc[twitter_mask, 'shares_count'].fillna(0) +
                        filtered_sentiment_processed.loc[twitter_mask, 'retweets_count'].fillna(0)
                    )
            
            # Define clean engagement metrics (without separate replies/retweets)
            engagement_cols = ['views_count', 'likes_count', 'shares_count', 'comments_count']
            available_engagement = [col for col in engagement_cols if col in filtered_sentiment_processed.columns]
            
            if available_engagement:
                # Engagement overview
                st.subheader("📊 Overview Engagement")
                
                cols = st.columns(len(available_engagement))
                for i, col in enumerate(available_engagement):
                    with cols[i]:
                        total = filtered_sentiment_processed[col].sum()
                        metric_name = col.replace('_count', '').title()
                        if col == 'shares_count':
                            # Add note for Twitter shares
                            st.metric(metric_name, f"{total:,}")
                            st.caption("Twitter: includes replies + retweets")
                        else:
                            st.metric(metric_name, f"{total:,}")
                
                # Engagement by platform
                st.subheader("📱 Engagement per Platform")
                
                platform_engagement = filtered_sentiment_processed.groupby('platform')[available_engagement].sum().reset_index()
                
                # Create subplots for each metric (max 4: Views, Likes, Shares, Comments)
                num_metrics = len(available_engagement)
                cols_per_row = 2
                rows_needed = (num_metrics + cols_per_row - 1) // cols_per_row
                
                subplot_titles = []
                for col in available_engagement:
                    title = col.replace('_count', '').title()
                    if col == 'shares_count':
                        title += " (Twitter: replies+retweets)"
                    subplot_titles.append(title)
                
                fig = make_subplots(
                    rows=rows_needed, 
                    cols=cols_per_row,
                    subplot_titles=subplot_titles
                )
                
                for i, col in enumerate(available_engagement):
                    row = (i // cols_per_row) + 1
                    col_pos = (i % cols_per_row) + 1
                    
                    fig.add_trace(
                        go.Bar(
                            x=platform_engagement['platform'],
                            y=platform_engagement[col],
                            name=col.replace('_count', '').title(),
                            showlegend=False
                        ),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(height=400, title_text="Engagement Metrics by Platform (Clean)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced engagement rate calculation
                if 'views_count' in available_engagement and 'likes_count' in available_engagement:
                    st.subheader("📈 Engagement Rate Analysis")
                    
                    # Ensure numeric columns
                    engagement_data = ensure_numeric_columns(
                        filtered_sentiment_processed, 
                        columns=available_engagement, 
                        default_value=1
                    )
                    
                    # Calculate comprehensive engagement rate (clean components)
                    engagement_components = [col for col in ['likes_count', 'shares_count', 'comments_count'] if col in engagement_data.columns]
                    
                    if engagement_components:
                        engagement_data['total_engagement'] = engagement_data[engagement_components].sum(axis=1)
                        engagement_data['engagement_rate'] = (
                            engagement_data['total_engagement'] / engagement_data['views_count'] * 100
                        ).fillna(0)
                        
                        # Cap extreme values
                        engagement_data['engagement_rate'] = engagement_data['engagement_rate'].clip(upper=100)
                        
                        # Platform comparison
                        avg_engagement = engagement_data.groupby('platform')['engagement_rate'].mean().reset_index()
                        platform_posts = engagement_data.groupby('platform').size().reset_index(name='post_count')
                        avg_engagement = pd.merge(avg_engagement, platform_posts, on='platform')
                        
                        # Enhanced visualization
                        fig = px.bar(
                            avg_engagement,
                            x='platform',
                            y='engagement_rate',
                            title="Average Engagement Rate by Platform (%) - Clean Formula<br><sub>Twitter shares = original shares + replies + retweets</sub>",
                            color='engagement_rate',
                            color_continuous_scale='viridis',
                            text=avg_engagement['engagement_rate'].round(2),
                            hover_data=['post_count']
                        )
                        
                        fig.update_traces(
                            texttemplate='%{text}%', 
                            textposition='outside'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Engagement distribution
                        st.subheader("📊 Distribusi Engagement Rate")
                        
                        fig = px.histogram(
                            engagement_data,
                            x='engagement_rate',
                            title="Distribusi Engagement Rate",
                            nbins=30,
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Top performing posts
                        st.subheader("🏆 Top Performing Posts")
                        
                        top_posts = engagement_data.nlargest(5, 'engagement_rate')[
                            ['platform', 'content_text_cleaned', 'engagement_rate', 'views_count', 'total_engagement']
                        ]
                        
                        for idx, post in top_posts.iterrows():
                            with st.expander(f"Post from {post['platform']} - {post['engagement_rate']:.2f}% engagement rate"):
                                st.write(f"**Content:** {post['content_text_cleaned'][:200]}...")
                                st.write(f"**Views:** {post['views_count']:,}")
                                st.write(f"**Total Engagement:** {post['total_engagement']:,}")
                                st.write(f"**Engagement Rate:** {post['engagement_rate']:.2f}%")
            
            else:
                st.info("Data engagement metrics tidak tersedia")
    
    with tab5:
        st.header("😊 Analisis Sentimen")
        
        if not filtered_sentiment.empty and 'sentiment' in filtered_sentiment.columns:
            # Sentiment overview
            st.subheader("📊 Overview Sentimen")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = filtered_sentiment['sentiment'].value_counts()
                dominant_sentiment = sentiment_counts.index[0]
                st.metric("Sentimen Dominan", dominant_sentiment)
            
            with col2:
                total_analyzed = len(filtered_sentiment)
                st.metric("Total Posts Analyzed", f"{total_analyzed:,}")
            
            # Sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Distribusi Sentimen",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        "Positive": "#4a90e2",
                        "Negative": "#e74c3c",
                        "Neutral": "#95a5a6"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment by platform
                sentiment_platform = filtered_sentiment.groupby(['platform', 'sentiment']).size().reset_index(name='count')
                fig = px.bar(
                    sentiment_platform,
                    x='platform',
                    y='count',
                    color='sentiment',
                    title="Sentimen per Platform",
                    barmode='stack'
                )
                st.plotly_chart(fig, use_container_width=True)
            

            
            # Enhanced sentiment trends
            st.subheader("📈 Tren Sentimen")
            
            filtered_sentiment['date'] = filtered_sentiment['timestamp'].dt.date
            sentiment_trend = filtered_sentiment.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            if not sentiment_trend.empty:
                fig = px.line(
                    sentiment_trend,
                    x='date',
                    y='count',
                    color='sentiment',
                    title="Tren Sentimen Harian"
                )
                st.plotly_chart(fig, use_container_width=True)
            

        
        else:
            st.info("Data sentimen tidak tersedia")
    
    with tab6:
        st.header("🎭 Analisis Emotion")
        
        if not filtered_sentiment.empty and 'emotion' in filtered_sentiment.columns:
            # Emotion overview
            st.subheader("📊 Overview Emotion")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                emotion_count = filtered_sentiment['emotion'].nunique()
                st.metric("Jenis Emotion", emotion_count)
            
            with col2:
                most_common_emotion = filtered_sentiment['emotion'].mode().iloc[0] if not filtered_sentiment['emotion'].mode().empty else "N/A"
                st.metric("Emotion Dominan", most_common_emotion)
            
            with col3:
                if 'views_count' in filtered_sentiment.columns:
                    emotion_engagement = filtered_sentiment.groupby('emotion')['views_count'].sum()
                    top_engagement_emotion = emotion_engagement.idxmax()
                    st.metric("Emotion Ter-engage", top_engagement_emotion)
            
            # Emotion distribution and confidence
            col1, col2 = st.columns(2)
            
            with col1:
                emotion_counts = filtered_sentiment['emotion'].value_counts()
                fig = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Distribusi Emotion",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Emotion by platform
                emotion_platform = filtered_sentiment.groupby(['platform', 'emotion']).size().reset_index(name='count')
                fig = px.bar(
                    emotion_platform,
                    x='platform',
                    y='count',
                    color='emotion',
                    title="Emotion per Platform",
                    barmode='stack',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Emotion vs Sentiment correlation with confidence
            if 'sentiment' in filtered_sentiment.columns:
                st.subheader("🔗 Emotion vs Sentiment Analysis")
                
                # Enhanced correlation matrix
                emotion_sentiment = pd.crosstab(
                    filtered_sentiment['emotion'], 
                    filtered_sentiment['sentiment'], 
                    normalize='index'
                ) * 100
                
                fig = px.imshow(
                    emotion_sentiment.values,
                    x=emotion_sentiment.columns,
                    y=emotion_sentiment.index,
                    title="Heatmap Emotion vs Sentiment (%)",
                    color_continuous_scale='RdYlBu_r',
                    text_auto='.1f'
                )
                st.plotly_chart(fig, use_container_width=True)
                

            
            # Emotion trending over time
            st.subheader("📅 Tren Emotion")
            
            filtered_sentiment['date'] = filtered_sentiment['timestamp'].dt.date
            emotion_trend = filtered_sentiment.groupby(['date', 'emotion']).size().reset_index(name='count')
            
            if not emotion_trend.empty:
                # Show only top emotions for clarity
                top_emotions = emotion_counts.head(5).index.tolist()
                emotion_trend_filtered = emotion_trend[emotion_trend['emotion'].isin(top_emotions)]
                
                fig = px.line(
                    emotion_trend_filtered,
                    x='date',
                    y='count',
                    color='emotion',
                    title="Tren Top 5 Emotion dari Waktu ke Waktu"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Data emotion tidak tersedia")
    
    with tab7:
        st.header("☁️ Word Cloud Analysis")
        
        if not filtered_sentiment.empty and 'content_text_cleaned' in filtered_sentiment.columns:
            # Enhanced sidebar controls
            st.sidebar.subheader("🎛️ WordCloud Settings")
            
            # Sentiment filter
            available_sentiments = filtered_sentiment['sentiment'].unique().tolist() if 'sentiment' in filtered_sentiment.columns else []
            selected_sentiment = st.sidebar.selectbox(
                "Filter Sentimen",
                options=['Semua'] + available_sentiments,
                index=0
            )
            
            # Emotion filter
            available_emotions = []
            if 'emotion' in filtered_sentiment.columns:
                available_emotions = filtered_sentiment['emotion'].unique().tolist()
                selected_emotion_wc = st.sidebar.selectbox(
                    "Filter Emotion",
                    options=['Semua'] + available_emotions,
                    index=0
                )
            else:
                selected_emotion_wc = 'Semua'
            
            # Platform filter
            selected_platform_wc = st.sidebar.selectbox(
                "Platform WordCloud",
                options=['Semua Platform'] + selected_platforms,
                index=0
            )
            
            # Province filter (NEW)
            if 'province' in filtered_sentiment.columns:
                available_provinces_wc = [p for p in filtered_sentiment['province'].unique() if p != 'unknown']
                selected_province_wc = st.sidebar.selectbox(
                    "Filter Provinsi",
                    options=['Semua Provinsi'] + available_provinces_wc,
                    index=0
                )
            else:
                selected_province_wc = 'Semua Provinsi'
            
            # Confidence threshold (NEW) - DISABLED per user request
            confidence_threshold_wc = 0.0  # Include all data
            
            # Apply filters
            wc_data = filtered_sentiment.copy()
            
            if selected_sentiment != 'Semua':
                wc_data = wc_data[wc_data['sentiment'] == selected_sentiment]
            
            if selected_emotion_wc != 'Semua' and 'emotion' in wc_data.columns:
                wc_data = wc_data[wc_data['emotion'] == selected_emotion_wc]
            
            if selected_platform_wc != 'Semua Platform':
                wc_data = wc_data[wc_data['platform'] == selected_platform_wc]
            
            if selected_province_wc != 'Semua Provinsi':
                wc_data = wc_data[wc_data['province'] == selected_province_wc]
            
            if 'sentiment_confidence' in wc_data.columns:
                wc_data = wc_data[wc_data['sentiment_confidence'] >= confidence_threshold_wc]
            
            # Generate title
            title_parts = []
            if selected_platform_wc != 'Semua Platform':
                title_parts.append(f"Platform {selected_platform_wc}")
            if selected_sentiment != 'Semua':
                title_parts.append(f"Sentimen {selected_sentiment}")
            if selected_emotion_wc != 'Semua':
                title_parts.append(f"Emotion {selected_emotion_wc}")
            if selected_province_wc != 'Semua Provinsi':
                title_parts.append(f"Provinsi {selected_province_wc}")
            if confidence_threshold_wc > 0:
                title_parts.append(f"Confidence ≥{confidence_threshold_wc}")
            
            title = f"Word Cloud - {' | '.join(title_parts)}" if title_parts else "Word Cloud - Semua Data"
            
            st.subheader(title)
            
            if not wc_data.empty:
                fig = create_wordcloud(wc_data['content_text_cleaned'], title)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada teks yang cukup untuk membuat word cloud")
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Posts Analyzed", len(wc_data))
                with col2:
                    if 'sentiment_confidence' in wc_data.columns:
                        avg_conf = wc_data['sentiment_confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                with col3:
                    total_chars = wc_data['content_text_cleaned'].str.len().sum()
                    st.metric("Total Characters", f"{total_chars:,}")
            
            else:
                st.info("Tidak ada data yang sesuai dengan filter")
            
            # Enhanced comparison wordclouds
            if len(available_sentiments) > 1:
                st.subheader("📊 Perbandingan WordCloud per Sentimen")
                
                cols = st.columns(len(available_sentiments))
                for i, sentiment in enumerate(available_sentiments):
                    sentiment_data = filtered_sentiment[filtered_sentiment['sentiment'] == sentiment]
                    
                    # Apply other filters
                    if selected_platform_wc != 'Semua Platform':
                        sentiment_data = sentiment_data[sentiment_data['platform'] == selected_platform_wc]
                    if selected_province_wc != 'Semua Provinsi':
                        sentiment_data = sentiment_data[sentiment_data['province'] == selected_province_wc]
                    if 'sentiment_confidence' in sentiment_data.columns:
                        sentiment_data = sentiment_data[sentiment_data['sentiment_confidence'] >= confidence_threshold_wc]
                    
                    with cols[i]:
                        st.write(f"**{sentiment}** ({len(sentiment_data)} posts)")
                        if not sentiment_data.empty:
                            fig = create_wordcloud(
                                sentiment_data['content_text_cleaned'], 
                                f"WordCloud - {sentiment}"
                            )
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.info(f"Tidak ada teks untuk {sentiment}")
                        else:
                            st.info(f"Tidak ada data untuk {sentiment}")
        
        else:
            st.info("Data teks tidak tersedia")
    
    with tab8:
        st.header("🕸️ Social Network Analysis")
        
        if not filtered_sna.empty:
            # Enhanced SNA filters
            st.subheader("🎛️ Filter Network Analysis")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                sna_platforms = filtered_sna['platform'].unique().tolist()
                selected_sna_platforms = st.multiselect(
                    "Platform SNA",
                    options=sna_platforms,
                    default=sna_platforms,
                    key="sna_platform_filter"
                )
            
            with col2:
                sna_sentiments = filtered_sna['sentiment'].unique().tolist() if 'sentiment' in filtered_sna.columns else []
                selected_sna_sentiments = st.multiselect(
                    "Sentimen SNA",
                    options=sna_sentiments,
                    default=sna_sentiments,
                    key="sna_sentiment_filter"
                )
            
            with col3:
                sna_emotions = filtered_sna['emotion'].unique().tolist() if 'emotion' in filtered_sna.columns else []
                selected_sna_emotions = st.multiselect(
                    "Emotion SNA",
                    options=sna_emotions,
                    default=sna_emotions,
                    key="sna_emotion_filter"
                )
            
            with col4:
                relation_types = filtered_sna['relation'].unique().tolist() if 'relation' in filtered_sna.columns else []
                selected_relations = st.multiselect(
                    "Jenis Relasi",
                    options=relation_types,
                    default=relation_types,
                    key="sna_relation_filter"
                )
            
            with col5:
                # Province filter for SNA (NEW)
                if 'province' in filtered_sna.columns:
                    sna_provinces = [p for p in filtered_sna['province'].unique() if p != 'unknown']
                    selected_sna_provinces = st.multiselect(
                        "Provinsi SNA",
                        options=['Semua'] + sna_provinces,
                        default=['Semua'],
                        key="sna_province_filter"
                    )
                else:
                    selected_sna_provinces = ['Semua']
            
            # Apply SNA filters
            sna_filtered = filtered_sna.copy()
            
            if selected_sna_platforms:
                sna_filtered = sna_filtered[sna_filtered['platform'].isin(selected_sna_platforms)]
            
            if selected_sna_sentiments and 'sentiment' in sna_filtered.columns:
                sna_filtered = sna_filtered[sna_filtered['sentiment'].isin(selected_sna_sentiments)]
            
            if selected_sna_emotions and 'emotion' in sna_filtered.columns:
                sna_filtered = sna_filtered[sna_filtered['emotion'].isin(selected_sna_emotions)]
            
            if selected_relations:
                sna_filtered = sna_filtered[sna_filtered['relation'].isin(selected_relations)]
            
            if 'Semua' not in selected_sna_provinces and 'province' in sna_filtered.columns:
                sna_filtered = sna_filtered[sna_filtered['province'].isin(selected_sna_provinces)]
            
            # Display filter status
            st.info(f"📊 Filter Aktif: {len(selected_sna_platforms)} platform, "
                   f"{len(selected_sna_sentiments)} sentimen, "
                   f"{len(selected_sna_emotions)} emotion, "
                   f"{len(selected_relations)} relasi, "
                   f"{'Semua provinsi' if 'Semua' in selected_sna_provinces else f'{len(selected_sna_provinces)} provinsi'}")
            
            if not sna_filtered.empty:
                # Network analysis
                G, top_influencers, network_fig = create_network_graph(sna_filtered)
                
                if G:
                    # Network statistics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Nodes", G.number_of_nodes())
                    
                    with col2:
                        st.metric("Total Edges", G.number_of_edges())
                    
                    with col3:
                        density = nx.density(G)
                        st.metric("Network Density", f"{density:.3f}")
                    
                    with col4:
                        st.metric("Data Points", len(sna_filtered))
                    
                    with col5:
                        if 'province' in sna_filtered.columns:
                            provinces_in_network = len(sna_filtered[sna_filtered['province'] != 'unknown']['province'].unique())
                            st.metric("Provinsi", provinces_in_network)
                    
                    # Network visualization
                    st.subheader("🕸️ Visualisasi Network")
                    
                    if network_fig:
                        st.pyplot(network_fig)
                    
                    # Enhanced network insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Top influencers
                        st.subheader("🏆 Top Influencers")
                        
                        if top_influencers:
                            influencer_df = pd.DataFrame(top_influencers[:10], columns=['User', 'Influence Score'])
                            
                            fig = px.bar(
                                influencer_df,
                                x='Influence Score',
                                y='User',
                                orientation='h',
                                title="Top 10 Influencers",
                                color='Influence Score',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Relation type distribution (grouped)
                        st.subheader("🔗 Distribusi Relasi")
                        
                        # Apply grouping to relations with platform context
                        sna_filtered['relation_grouped'] = sna_filtered.apply(
                            lambda row: group_sna_relations(row['relation'], row.get('platform')), axis=1
                        )
                        relation_counts = sna_filtered['relation_grouped'].value_counts()
                        
                        fig = px.pie(
                            values=relation_counts.values,
                            names=relation_counts.index,
                            title="Jenis Relasi (Grouped)",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed breakdown
                        st.write("**Detail Grouping:**")
                        for group, count in relation_counts.items():
                            percentage = (count / relation_counts.sum()) * 100
                            st.write(f"• {group}: {count:,} ({percentage:.1f}%)")
                        
                        # Option to show original relation details
                        if st.checkbox("🔍 Tampilkan Detail Relasi Asli", key="show_original_relations"):
                            st.write("**Relasi Detail (Top 15):**")
                            original_relation_counts = sna_filtered['relation'].value_counts().head(15)
                            
                            for relation, count in original_relation_counts.items():
                                # Get most common platform for this relation for grouping reference
                                platform_for_relation = sna_filtered[sna_filtered['relation'] == relation]['platform'].mode()
                                platform_ref = platform_for_relation.iloc[0] if not platform_for_relation.empty else None
                                grouped = group_sna_relations(relation, platform_ref)
                                percentage = (count / len(sna_filtered)) * 100
                                st.write(f"• {relation} → _{grouped}_: {count:,} ({percentage:.1f}%)")
                    
                    # Provincial network analysis (NEW)
                    if 'province' in sna_filtered.columns and 'Semua' not in selected_sna_provinces:
                        st.subheader("🗺️ Analisis Network per Provinsi")
                        
                        province_network_stats = []
                        for province in selected_sna_provinces:
                            if province != 'Semua':
                                province_sna = sna_filtered[sna_filtered['province'] == province]
                                if not province_sna.empty:
                                    province_G, _, _ = create_network_graph(province_sna)
                                    if province_G:
                                        province_network_stats.append({
                                            'Provinsi': province,
                                            'Nodes': province_G.number_of_nodes(),
                                            'Edges': province_G.number_of_edges(),
                                            'Density': nx.density(province_G),
                                            'Data Points': len(province_sna)
                                        })
                        
                        if province_network_stats:
                            province_df = pd.DataFrame(province_network_stats)
                            st.dataframe(province_df, use_container_width=True)
                            
                            # Visualization
                            fig = px.bar(
                                province_df,
                                x='Provinsi',
                                y='Nodes',
                                title="Jumlah Nodes per Provinsi",
                                color='Density',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("Tidak ada data network yang cukup untuk analisis")
            
            else:
                st.warning("Tidak ada data yang sesuai dengan filter")
        
        else:
            st.info("Data SNA tidak tersedia")
    
    with tab9:
        st.header("🔥 Trending Topics & Keywords")
        
        # KeyBERT info
        with st.expander("ℹ️ Tentang Enhanced Topic Analysis", expanded=False):
            st.write("""
            **🚀 Enhanced Topic Analysis Features:**
            
            - **KeyBERT Integration**: Semantic topic extraction
            - **Confidence Filtering**: Topics dari data dengan confidence tinggi
            - **Geographic Analysis**: Trending topics per provinsi  
            - **Entity-based Topics**: Topics yang melibatkan specific entities
            - **Cross-platform Comparison**: Trending topics analysis antar platform
            """)
        
        if not filtered_sentiment.empty and 'content_text_cleaned' in filtered_sentiment.columns:
            # High confidence topics
            st.subheader("🎯 Trending Topics (High Confidence)")
            
            high_conf_data = filtered_sentiment.copy()
            if 'sentiment_confidence' in high_conf_data.columns:
                high_conf_data = high_conf_data[high_conf_data['sentiment_confidence'] >= 0.8]
            
            with st.spinner("🤖 Mengekstrak trending topics..."):
                topics = extract_topics(high_conf_data['content_text_cleaned'], n_topics=15)
            
            if topics:
                topics_df = pd.DataFrame(topics, columns=['Topic', 'Relevance Score'])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        topics_df.head(10),
                        x='Relevance Score',
                        y='Topic',
                        orientation='h',
                        title="Top 10 Trending Topics (High Confidence Data)",
                        color='Relevance Score',
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Analisis Detail:**")
                    selected_topic = st.selectbox(
                        "Pilih Topic",
                        options=[None] + [topic for topic, _ in topics[:15]],
                        index=0
                    )
                
                # Enhanced topic analysis
                if selected_topic:
                    st.subheader(f"📊 Deep Dive: '{selected_topic}'")
                    
                    # Smart topic filtering - split topic into words and find posts containing any of them
                    topic_words = [word.strip().lower() for word in selected_topic.split() if len(word.strip()) > 2]
                    
                    if topic_words:
                        # Create regex pattern to find posts containing at least 2 words from the topic (for multi-word topics)
                        # or 1 word for single-word topics
                        min_words_required = 1 if len(topic_words) == 1 else 2
                        
                        topic_posts_list = []
                        for idx, row in filtered_sentiment.iterrows():
                            content = str(row.get('content_text_cleaned', '')).lower()
                            matches = sum(1 for word in topic_words if word in content)
                            
                            if matches >= min_words_required:
                                topic_posts_list.append(idx)
                        
                        topic_posts = filtered_sentiment.loc[topic_posts_list] if topic_posts_list else pd.DataFrame()
                        
                        # Debug info
                        st.caption(f"🔍 Searching for posts containing at least {min_words_required} words from: {', '.join(topic_words)}")
                    else:
                        # Fallback to original method if word parsing fails
                        topic_posts = filtered_sentiment[
                            filtered_sentiment['content_text_cleaned'].str.contains(
                                selected_topic, case=False, na=False
                            )
                        ]
                    
                    if not topic_posts.empty:
                        # Enhanced metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Posts", len(topic_posts))
                        
                        with col2:
                            if 'views_count' in topic_posts.columns:
                                total_views = topic_posts['views_count'].sum()
                                st.metric("Total Views", f"{total_views:,}")
                        
                        with col3:
                            platforms_count = topic_posts['platform'].nunique()
                            st.metric("Platforms", platforms_count)
                        
                        with col4:
                            if 'province' in topic_posts.columns:
                                provinces_count = len(topic_posts[topic_posts['province'] != 'unknown']['province'].unique())
                                st.metric("Provinsi", provinces_count)
                        
                        # Trend Sentiment Analysis for Selected Topic
                        if 'sentiment' in topic_posts.columns and 'date' in topic_posts.columns:
                            st.subheader(f"📈 Tren Sentimen: '{selected_topic}'")
                            
                            # Prepare data for trend analysis
                            topic_posts_copy = topic_posts.copy()
                            
                            # Ensure date column is datetime
                            try:
                                topic_posts_copy['date'] = pd.to_datetime(topic_posts_copy['date'])
                                topic_posts_copy['date_only'] = topic_posts_copy['date'].dt.date
                                
                                # Group by date and sentiment
                                sentiment_trend = topic_posts_copy.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
                                sentiment_trend['date_only'] = pd.to_datetime(sentiment_trend['date_only'])
                                
                                if not sentiment_trend.empty and len(sentiment_trend) > 1:
                                    # Create trend visualization
                                    fig = px.line(
                                        sentiment_trend,
                                        x='date_only',
                                        y='count',
                                        color='sentiment',
                                        title=f"Tren Sentimen Harian untuk Topic: '{selected_topic}'",
                                        labels={'date_only': 'Tanggal', 'count': 'Jumlah Posts', 'sentiment': 'Sentimen'},
                                        color_discrete_map={
                                            'positive': '#2E8B57',  # Green
                                            'negative': '#DC143C',  # Red
                                            'neutral': '#4682B4'    # Blue
                                        }
                                    )
                                    
                                    fig.update_layout(
                                        height=400,
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        )
                                    )
                                    
                                    fig.update_traces(mode='lines+markers', marker=dict(size=4))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Additional trend metrics
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        # Most positive day
                                        positive_trend = sentiment_trend[sentiment_trend['sentiment'] == 'positive']
                                        if not positive_trend.empty:
                                            max_positive_day = positive_trend.loc[positive_trend['count'].idxmax()]
                                            st.metric(
                                                "📅 Hari Paling Positif", 
                                                max_positive_day['date_only'].strftime('%d %b %Y'),
                                                f"{max_positive_day['count']} posts"
                                            )
                                    
                                    with col2:
                                        # Most negative day
                                        negative_trend = sentiment_trend[sentiment_trend['sentiment'] == 'negative']
                                        if not negative_trend.empty:
                                            max_negative_day = negative_trend.loc[negative_trend['count'].idxmax()]
                                            st.metric(
                                                "📅 Hari Paling Negatif", 
                                                max_negative_day['date_only'].strftime('%d %b %Y'),
                                                f"{max_negative_day['count']} posts"
                                            )
                                    
                                    with col3:
                                        # Overall sentiment distribution for this topic
                                        topic_sentiment_dist = topic_posts['sentiment'].value_counts()
                                        if not topic_sentiment_dist.empty:
                                            dominant_sentiment = topic_sentiment_dist.index[0]
                                            percentage = (topic_sentiment_dist.iloc[0] / len(topic_posts)) * 100
                                            st.metric(
                                                "🎯 Sentimen Dominan", 
                                                dominant_sentiment.title(),
                                                f"{percentage:.1f}%"
                                            )
                                else:
                                    st.info("Data tren sentimen tidak cukup untuk ditampilkan (perlu minimal 2 data point)")
                                    
                            except Exception as e:
                                st.warning(f"Tidak dapat membuat tren sentimen: {str(e)}")
                        
                        # Geographic distribution of topic
                        if 'province' in topic_posts.columns:
                            st.subheader("🗺️ Distribusi Geografis Topic")
                            
                            topic_provinces = topic_posts[topic_posts['province'] != 'unknown']['province'].value_counts().head(10)
                            
                            if not topic_provinces.empty:
                                fig = px.bar(
                                    x=topic_provinces.values,
                                    y=topic_provinces.index,
                                    orientation='h',
                                    title=f"Top Provinsi membahas '{selected_topic}'",
                                    color=topic_provinces.values,
                                    color_continuous_scale='viridis'
                                )
                                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Entity analysis for topic
                        if any(col in topic_posts.columns for col in ['PER_parsed', 'ORG_parsed', 'INFLUENCER_parsed']):
                            st.subheader("🤖 Entities terkait Topic")
                            
                            entity_cols = st.columns(3)
                            
                            for i, col_name in enumerate(['PER_parsed', 'ORG_parsed', 'INFLUENCER_parsed']):
                                if col_name in topic_posts.columns:
                                    with entity_cols[i]:
                                        entity_type = col_name.replace('_parsed', '')
                                        st.write(f"**{entity_type}:**")
                                        
                                        all_entities = []
                                        for entities in topic_posts[col_name].dropna():
                                            if isinstance(entities, list):
                                                all_entities.extend(entities)
                                        
                                        if all_entities:
                                            entity_counts = Counter(all_entities)
                                            top_entities = entity_counts.most_common(5)
                                            
                                            for entity, count in top_entities:
                                                st.write(f"• {entity}: {count}")
                        

            
            # Provincial trending topics (NEW)
            if 'province' in filtered_sentiment.columns:
                st.subheader("🗺️ Trending Topics per Provinsi")
                
                # Get top provinces
                top_provinces = filtered_sentiment[filtered_sentiment['province'] != 'unknown']['province'].value_counts().head(6).index.tolist()
                
                if top_provinces:
                    province_cols = st.columns(min(3, len(top_provinces)))
                    
                    for i, province in enumerate(top_provinces):
                        province_data = filtered_sentiment[filtered_sentiment['province'] == province]
                        
                        with province_cols[i % len(province_cols)]:
                            st.write(f"**{province}**")
                            
                            if not province_data.empty:
                                province_topics = extract_topics(province_data['content_text_cleaned'], n_topics=5)
                                
                                if province_topics:
                                    topics_df = pd.DataFrame(province_topics, columns=['Topic', 'Score'])
                                    st.dataframe(topics_df, use_container_width=True, height=200)
                                else:
                                    st.info(f"Tidak ada topics untuk {province}")
        
        else:
            st.info("Data tidak tersedia untuk analisis trending")
    


if __name__ == "__main__":
    main() 