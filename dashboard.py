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
warnings.filterwarnings('ignore')

# Initialize KeyBERT model
@st.cache_resource
def load_keybert_model():
    """Load and cache KeyBERT model"""
    try:
        # Use a lightweight multilingual model that works well for Indonesian
        # First time loading might take a few seconds to download the model
        kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
        return kw_model
    except Exception as e:
        st.error(f"❌ Error loading KeyBERT model: {str(e)}")
        st.info("💡 Tip: Pastikan koneksi internet stabil untuk download model pertama kali.")
        return None

# Helper function to ensure numeric columns have valid values for calculation
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
            if col in ['views'] and (result[col] == 0).any():
                result[col] = result[col].replace(0, 1)
                
    return result

# Page configuration
st.set_page_config(
    page_title="Dashboard Analisis Media Sosial",
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        # Load sentiment data
        sentiment_df = pd.read_csv('dashboardsentimen_with_sentiment.csv')
        
        # Load SNA data
        sna_df = pd.read_csv('dashboardsna_with_sentiment.csv')
        
        # Convert timestamp columns
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        sna_df['timestamp'] = pd.to_datetime(sna_df['timestamp'])
        
        return sentiment_df, sna_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def extract_topics(text_series, n_topics=10):
    """Extract top topics from text data using KeyBERT for semantic topic extraction"""
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
        # Use keyphrase_ngram_range to get both single words and phrases
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3),  # Extract 1-3 word phrases
            stop_words='english',  # Basic English stopwords
            use_maxsum=True,  # Use MaxSum for diversity
            nr_candidates=20,  # Consider top 20 candidates
            diversity=0.5  # Balance between relevance and diversity
        )[:n_topics]  # Take top n_topics manually
        
        # Convert to the same format as the old function (topic, score)
        # KeyBERT returns (keyword, score) tuples
        topics = [(keyword, round(score * 100)) for keyword, score in keywords]
        
        # Filter out very short keywords and common Indonesian/English words
        indonesian_stopwords = {'yang', 'dan', 'untuk', 'dari', 'dengan', 'pada', 'ini', 'itu', 'adalah', 'akan', 'sudah', 'tidak', 'ada', 'juga', 'bisa', 'saya', 'kita', 'mereka', 'dalam', 'atau', 'the', 'and', 'or', 'to', 'of', 'in', 'for', 'is', 'are', 'was', 'were', 'have', 'has', 'can', 'will', 'would', 'could', 'should'}
        
        filtered_topics = []
        for topic, score in topics:
            # Filter out stopwords and very short topics
            topic_words = topic.lower().split()
            if (len(topic.strip()) >= 3 and 
                not any(word in indonesian_stopwords for word in topic_words) and
                not topic.lower().isdigit()):
                filtered_topics.append((topic, score))
        
        return filtered_topics[:n_topics] if filtered_topics else extract_topics_fallback(text_series, n_topics)
    
    except Exception as e:
        st.warning(f"⚠️ KeyBERT extraction failed: {str(e)}. Menggunakan metode fallback (frequency-based).")
        return extract_topics_fallback(text_series, n_topics)

def extract_topics_fallback(text_series, n_topics=10):
    """Fallback topic extraction using word frequency (original method)"""
    if text_series.empty:
        return []
    
    # Combine all text
    text = ' '.join(text_series.fillna('').astype(str))
    
    # Simple topic extraction using word frequency
    # Remove mentions, hashtags, URLs
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
    
    # Top influencers (nodes with highest weighted in-degree)
    top_influencers = sorted(weighted_in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Create network visualization
    network_fig = create_network_visualization(G)
    
    return G, top_influencers, network_fig

def create_network_visualization(G):
    """Create network visualization using matplotlib with weighted edges and sized nodes"""
    if G.number_of_nodes() == 0:
        return None
    
    # Limit nodes for better visualization
    if G.number_of_nodes() > 50:
        # Take only top nodes by weighted degree
        weighted_degrees = dict(G.degree(weight='weight'))
        top_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:50]
        top_node_names = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_names)
    
    # Create layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=100, weight='weight')
    except:
        pos = nx.random_layout(G)
    
    # Calculate node sizes based on weighted in-degree (influence)
    weighted_in_degrees = dict(G.in_degree(weight='weight'))
    max_weighted_in_degree = max(weighted_in_degrees.values()) if weighted_in_degrees.values() else 1
    min_weighted_in_degree = min(weighted_in_degrees.values()) if weighted_in_degrees.values() else 0
    
    # Scale node sizes between 200 and 2000
    node_sizes = []
    for node in G.nodes():
        if max_weighted_in_degree > min_weighted_in_degree:
            normalized = (weighted_in_degrees[node] - min_weighted_in_degree) / (max_weighted_in_degree - min_weighted_in_degree)
            size = 200 + (normalized * 1800)  # Between 200 and 2000
        else:
            size = 600  # Default size if all nodes have same degree
        node_sizes.append(size)
    
    # Calculate node colors based on weighted in-degree
    node_colors = [weighted_in_degrees[node] / max_weighted_in_degree for node in G.nodes()]
    
    # Calculate edge widths based on edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    min_edge_weight = min(edge_weights) if edge_weights else 1
    
    # Scale edge widths between 0.5 and 5
    edge_widths = []
    for weight in edge_weights:
        if max_edge_weight > min_edge_weight:
            normalized = (weight - min_edge_weight) / (max_edge_weight - min_edge_weight)
            width = 0.5 + (normalized * 4.5)  # Between 0.5 and 5
        else:
            width = 2  # Default width if all edges have same weight
        edge_widths.append(width)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges with varying thickness
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          alpha=0.6,
                          arrows=True,
                          arrowsize=15,
                          arrowstyle='-|>',
                          width=edge_widths,
                          ax=ax)
    
    # Draw nodes with varying sizes
    nodes = nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors,
                          cmap=plt.cm.plasma,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1,
                          ax=ax)
    
    # Add labels for top nodes only
    weighted_degrees = dict(G.degree(weight='weight'))
    top_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
    top_node_names = [node for node, _ in top_nodes]
    labels = {node: node[:15] + '...' if len(node) > 15 else node for node in top_node_names}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title("Network Visualization\n(Node size = Influence, Edge thickness = Relationship strength)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar for node influence
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                              norm=plt.Normalize(vmin=min_weighted_in_degree, vmax=max_weighted_in_degree))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Weighted In-Degree (Influence Level)', rotation=270, labelpad=20)
    
    # Add legend for edge thickness
    legend_text = f"Edge Thickness: Min={min_edge_weight}, Max={max_edge_weight} connections"
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Dashboard Analisis Media Sosial</h1>', unsafe_allow_html=True)
    
    # Load data
    sentiment_df, sna_df = load_data()
    
    if sentiment_df is None or sna_df is None:
        st.error("Gagal memuat data. Pastikan file CSV tersedia.")
        return
    
    # Sidebar for filters
    st.sidebar.header("🔧 Filter & Pengaturan")
    
    # Platform filter
    all_platforms_sentiment = sentiment_df['platform'].unique().tolist() if 'platform' in sentiment_df.columns else []
    all_platforms_sna = sna_df['platform'].unique().tolist() if 'platform' in sna_df.columns else []
    all_platforms = list(set(all_platforms_sentiment + all_platforms_sna))
    
    selected_platforms = st.sidebar.multiselect(
        "Pilih Platform",
        options=all_platforms,
        default=all_platforms[:3] if len(all_platforms) >= 3 else all_platforms
    )
    
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
    
    # Filter data based on selection
    if selected_platforms:
        filtered_sentiment = sentiment_df[sentiment_df['platform'].isin(selected_platforms)]
        filtered_sna = sna_df[sna_df['platform'].isin(selected_platforms)]
    else:
        filtered_sentiment = sentiment_df
        filtered_sna = sna_df
    
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", "🎯 Engagement", "😊 Sentimen", "🎭 Emotion", "☁️ WordCloud", "🕸️ SNA", "🔥 Trending"
    ])
    
    with tab1:
        st.header("📈 Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_posts = len(filtered_sentiment)
            st.metric("Total Posts", f"{total_posts:,}")
        
        with col2:
            if 'views' in filtered_sentiment.columns:
                total_views = filtered_sentiment['views'].sum()
                st.metric("Total Views", f"{total_views:,}")
        
        with col3:
            if 'likes' in filtered_sentiment.columns:
                total_likes = filtered_sentiment['likes'].sum()
                st.metric("Total Likes", f"{total_likes:,}")
        
        with col4:
            platforms_count = len(filtered_sentiment['platform'].unique())
            st.metric("Platforms", platforms_count)
        
        # Platform distribution
        if not filtered_sentiment.empty:
            st.subheader("Distribusi Platform")
            platform_counts = filtered_sentiment['platform'].value_counts()
            
            fig = px.pie(
                values=platform_counts.values,
                names=platform_counts.index,
                title="Distribusi Posts per Platform"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🎯 Analisis Engagement")
        
        if not filtered_sentiment.empty:
            # Check if all required columns are available
            has_engagement_metrics = all(col in filtered_sentiment.columns for col in ['views', 'likes', 'shares', 'comments'])
            
            if not has_engagement_metrics:
                st.warning("⚠️ Beberapa kolom engagement metrics (views, likes, shares, comments) mungkin tidak tersedia di semua platform. Dashboard akan menampilkan data yang tersedia.")
                
                # Create default columns if needed
                for col in ['views', 'likes', 'shares', 'comments']:
                    if col not in filtered_sentiment.columns:
                        filtered_sentiment[col] = 0
            
            # Engagement metrics by platform
            st.subheader("Metrics Engagement per Platform")
            
            engagement_metrics = filtered_sentiment.groupby('platform').agg({
                'views': 'sum',
                'likes': 'sum',
                'shares': 'sum',
                'comments': 'sum'
            }).reset_index()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Views', 'Likes', 'Shares', 'Comments'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            metrics = ['views', 'likes', 'shares', 'comments']
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for metric, (row, col) in zip(metrics, positions):
                fig.add_trace(
                    go.Bar(
                        x=engagement_metrics['platform'],
                        y=engagement_metrics[metric],
                        name=metric.capitalize(),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Engagement Metrics by Platform")
            st.plotly_chart(fig, use_container_width=True)
            
            # Engagement rate calculation
            st.subheader("Engagement Rate")
            
            # Process data to ensure all required columns exist and have valid values
            engagement_data = ensure_numeric_columns(
                filtered_sentiment, 
                columns=['views', 'likes', 'shares', 'comments'], 
                default_value=1  # Use 1 for views to avoid division by zero
            )
            
            # Calculate engagement rate with the formula: (likes + comments + shares) / views * 100%
            engagement_data['engagement_rate'] = (
                (engagement_data['likes'] + engagement_data['shares'] + engagement_data['comments']) / 
                engagement_data['views'] * 100
            ).fillna(0)
            
            # For platforms with very high engagement_rate (probably due to low/missing views),
            # cap the values to make visualization more meaningful
            engagement_data['engagement_rate'] = engagement_data['engagement_rate'].clip(upper=100)
            
            # Add platform-specific notes for context
            st.info("💡 **Formula Engagement Rate**: (likes + comments + shares) / views × 100%\n\n"
                  "Nilai engagement rate menunjukkan seberapa aktif pengguna berinteraksi dengan konten. "
                  "Semakin tinggi persentase, semakin tinggi interaksi pengguna terhadap jumlah view.")
            
            # Calculate average engagement rate by platform
            avg_engagement = engagement_data.groupby('platform')['engagement_rate'].mean().reset_index()
            
            # Add count of posts per platform for context
            platform_posts = engagement_data.groupby('platform').size().reset_index(name='post_count')
            avg_engagement = pd.merge(avg_engagement, platform_posts, on='platform')
            
            # Add post count to hover info
            hover_template = (
                '<b>%{x}</b><br>' +
                'Engagement Rate: %{y:.2f}%<br>' +
                'Posts: %{customdata[0]}<br>'
            )
            
            # Enhanced visualization with more information
            fig = px.bar(
                avg_engagement,
                x='platform',
                y='engagement_rate',
                title="Average Engagement Rate by Platform (%) - Formula: (likes+comments+shares)/views*100%",
                color='engagement_rate',
                color_continuous_scale='viridis',
                text=avg_engagement['engagement_rate'].round(2),
                custom_data=[avg_engagement['post_count']]  # Include post count for hover
            )
            
            # Improve readability of the chart
            fig.update_traces(
                texttemplate='%{text}%', 
                textposition='outside',
                hovertemplate=hover_template
            )
            fig.update_layout(
                uniformtext_minsize=8, 
                uniformtext_mode='hide',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("😊 Analisis Sentimen")
        
        if not filtered_sentiment.empty and 'sentiment' in filtered_sentiment.columns:
            # Sentiment distribution
            st.subheader("Distribusi Sentimen")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = filtered_sentiment['sentiment'].value_counts()

                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Distribusi Sentimen Keseluruhan",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        "Positif": "#4a90e2",   # biru elegan
                        "Negatif": "#e74c3c",   # merah elegan
                        "Netral": "#95a5a6"     # abu-abu netral
                    } 
                )

                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
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
            
            # Sentiment trend over time
            st.subheader("Tren Sentimen dari Waktu ke Waktu")
            
            filtered_sentiment['date'] = filtered_sentiment['timestamp'].dt.date
            sentiment_trend = filtered_sentiment.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            fig = px.line(
                sentiment_trend,
                x='date',
                y='count',
                color='sentiment',
                title="Tren Sentimen Harian"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Platform comparison
            if len(selected_platforms) > 1:
                st.subheader("Perbandingan Sentimen Antar Platform")
                
                comparison_data = []
                for platform in selected_platforms:
                    platform_data = filtered_sentiment[filtered_sentiment['platform'] == platform]
                    sentiment_pct = platform_data['sentiment'].value_counts(normalize=True) * 100
                    
                    for sentiment, percentage in sentiment_pct.items():
                        comparison_data.append({
                            'Platform': platform,
                            'Sentiment': sentiment,
                            'Percentage': percentage
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig = px.bar(
                    comparison_df,
                    x='Platform',
                    y='Percentage',
                    color='Sentiment',
                    title="Perbandingan Persentase Sentimen Antar Platform",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Quick emotion insight in sentiment tab
            if 'emotion' in filtered_sentiment.columns:
                st.subheader("💡 Quick Emotion Insight")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Most correlated emotion-sentiment pairs
                    emotion_sentiment_corr = filtered_sentiment.groupby(['emotion', 'sentiment']).size().reset_index(name='count')
                    top_emotion_sentiment = emotion_sentiment_corr.nlargest(5, 'count')
                    
                    st.write("**Top 5 Kombinasi Emotion-Sentiment:**")
                    for _, row in top_emotion_sentiment.iterrows():
                        st.write(f"• {row['emotion']} + {row['sentiment']}: {row['count']} posts")
                
                with col2:
                    # Emotion diversity by platform
                    if len(selected_platforms) > 1:
                        emotion_diversity = []
                        for platform in selected_platforms:
                            platform_data = filtered_sentiment[filtered_sentiment['platform'] == platform]
                            unique_emotions = platform_data['emotion'].nunique()
                            emotion_diversity.append({
                                'Platform': platform,
                                'Unique Emotions': unique_emotions
                            })
                        
                        emotion_div_df = pd.DataFrame(emotion_diversity)
                        
                        st.write("**Keragaman Emotion per Platform:**")
                        for _, row in emotion_div_df.iterrows():
                            st.write(f"• {row['Platform']}: {row['Unique Emotions']} jenis emotion")
                
                # Call-to-action
                st.info("💡 **Tip:** Lihat tab 'Emotion' untuk analisis emotion yang lebih detail!")
    
    with tab4:
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
                if 'views' in filtered_sentiment.columns:
                    emotion_engagement = filtered_sentiment.groupby('emotion')['views'].sum()
                    top_engagement_emotion = emotion_engagement.idxmax()
                    st.metric("Emotion Ter-engage", top_engagement_emotion)
            
            # Emotion distribution
            st.subheader("📈 Distribusi Emotion")
            
            col1, col2 = st.columns(2)
            
            with col1:
                emotion_counts = filtered_sentiment['emotion'].value_counts()
                fig = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Distribusi Emotion Keseluruhan",
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
            
            # Emotion vs Sentiment Correlation
            st.subheader("🔗 Korelasi Emotion vs Sentiment")
            
            if 'sentiment' in filtered_sentiment.columns:
                # Cross-tabulation
                emotion_sentiment = pd.crosstab(
                    filtered_sentiment['emotion'], 
                    filtered_sentiment['sentiment'], 
                    normalize='index'
                ) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Heatmap
                    fig = px.imshow(
                        emotion_sentiment.values,
                        x=emotion_sentiment.columns,
                        y=emotion_sentiment.index,
                        title="Heatmap Emotion vs Sentiment (%)",
                        color_continuous_scale='RdYlBu_r',
                        text_auto='.1f'
                    )
                    fig.update_layout(
                        xaxis_title="Sentiment",
                        yaxis_title="Emotion"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Grouped bar chart
                    emotion_sentiment_df = filtered_sentiment.groupby(['emotion', 'sentiment']).size().reset_index(name='count')
                    fig = px.bar(
                        emotion_sentiment_df,
                        x='emotion',
                        y='count',
                        color='sentiment',
                        title="Distribusi Sentiment dalam setiap Emotion",
                        barmode='group'
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Emotion trend over time
            st.subheader("📅 Tren Emotion dari Waktu ke Waktu")
            
            filtered_sentiment['date'] = filtered_sentiment['timestamp'].dt.date
            emotion_trend = filtered_sentiment.groupby(['date', 'emotion']).size().reset_index(name='count')
            emotion_trend['date'] = pd.to_datetime(emotion_trend['date'])
            
            if len(emotion_trend) > 1:
                fig = px.line(
                    emotion_trend,
                    x='date',
                    y='count',
                    color='emotion',
                    title="Tren Emotion Harian",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Posts")
                st.plotly_chart(fig, use_container_width=True)
            
            # Emotion engagement analysis
            if all(col in filtered_sentiment.columns for col in ['views', 'likes', 'shares', 'comments']):
                st.subheader("💡 Analisis Engagement per Emotion")
                
                engagement_by_emotion = filtered_sentiment.groupby('emotion').agg({
                    'views': 'mean',
                    'likes': 'mean',
                    'shares': 'mean',
                    'comments': 'mean',
                    'content_text': 'count'
                }).round(2)
                
                engagement_by_emotion.columns = ['Avg Views', 'Avg Likes', 'Avg Shares', 'Avg Comments', 'Total Posts']
                engagement_by_emotion = engagement_by_emotion.sort_values('Avg Views', ascending=False)
                
                st.dataframe(engagement_by_emotion, use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=engagement_by_emotion.index,
                        y=engagement_by_emotion['Avg Views'],
                        title="Average Views per Emotion",
                        color=engagement_by_emotion['Avg Views'],
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Calculate engagement rate
                    engagement_by_emotion['Engagement Rate'] = (
                        (engagement_by_emotion['Avg Likes'] + 
                         engagement_by_emotion['Avg Shares'] + 
                         engagement_by_emotion['Avg Comments']) / 
                        engagement_by_emotion['Avg Views'] * 100
                    ).fillna(0)
                    
                    fig = px.bar(
                        x=engagement_by_emotion.index,
                        y=engagement_by_emotion['Engagement Rate'],
                        title="Engagement Rate per Emotion (%)",
                        color=engagement_by_emotion['Engagement Rate'],
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Platform comparison for emotions
            if len(selected_platforms) > 1:
                st.subheader("🏢 Perbandingan Emotion Antar Platform")
                
                platform_emotion_comparison = []
                for platform in selected_platforms:
                    platform_data = filtered_sentiment[filtered_sentiment['platform'] == platform]
                    emotion_pct = platform_data['emotion'].value_counts(normalize=True) * 100
                    
                    for emotion, percentage in emotion_pct.items():
                        platform_emotion_comparison.append({
                            'Platform': platform,
                            'Emotion': emotion,
                            'Percentage': percentage
                        })
                
                if platform_emotion_comparison:
                    comparison_df = pd.DataFrame(platform_emotion_comparison)
                    
                    fig = px.bar(
                        comparison_df,
                        x='Platform',
                        y='Percentage',
                        color='Emotion',
                        title="Perbandingan Persentase Emotion Antar Platform",
                        barmode='group',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Emotion-based content insights
            st.subheader("📝 Insight Konten per Emotion")
            
            selected_emotion = st.selectbox(
                "Pilih Emotion untuk Analisis Detail",
                options=filtered_sentiment['emotion'].unique().tolist(),
                key="emotion_selector"
            )
            
            if selected_emotion:
                emotion_data = filtered_sentiment[filtered_sentiment['emotion'] == selected_emotion]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Posts", len(emotion_data))
                
                with col2:
                    if 'views' in emotion_data.columns:
                        total_views = emotion_data['views'].sum()
                        st.metric("Total Views", f"{total_views:,}")
                
                with col3:
                    platforms_count = emotion_data['platform'].nunique()
                    st.metric("Platforms", platforms_count)
                
                # Sentiment distribution for selected emotion
                if 'sentiment' in emotion_data.columns:
                    sentiment_in_emotion = emotion_data['sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_in_emotion.values,
                        names=sentiment_in_emotion.index,
                        title=f"Distribusi Sentiment dalam Emotion '{selected_emotion}'"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sample posts for selected emotion
                st.write(f"**Sample Posts dengan Emotion '{selected_emotion}':**")
                
                if 'views' in emotion_data.columns:
                    sample_posts = emotion_data.nlargest(3, 'views')[['platform', 'content_text', 'sentiment', 'views', 'timestamp']]
                else:
                    sample_posts = emotion_data.head(3)[['platform', 'content_text', 'sentiment', 'timestamp']]
                
                for idx, post in sample_posts.iterrows():
                    with st.expander(f"Post from {post['platform']} - {post['sentiment']} sentiment"):
                        st.write(f"**Content:** {post['content_text'][:200]}...")
                        st.write(f"**Date:** {post['timestamp']}")
                        if 'views' in post:
                            st.write(f"**Views:** {post['views']:,}")
        
        else:
            st.info("Data emotion tidak tersedia dalam dataset yang dipilih")
    
    with tab5:
        st.header("☁️ Word Cloud")
        
        if not filtered_sentiment.empty and 'content_text' in filtered_sentiment.columns and 'sentiment' in filtered_sentiment.columns:
            # Sidebar controls for wordcloud
            st.sidebar.subheader("🎛️ WordCloud Settings")
            
            # Sentiment filter
            available_sentiments = filtered_sentiment['sentiment'].unique().tolist()
            selected_sentiment = st.sidebar.selectbox(
                "Filter berdasarkan Sentimen",
                options=['Semua'] + available_sentiments,
                index=0
            )
            
            # Emotion filter
            available_emotions = []
            if 'emotion' in filtered_sentiment.columns:
                available_emotions = filtered_sentiment['emotion'].unique().tolist()
                selected_emotion_wc = st.sidebar.selectbox(
                    "Filter berdasarkan Emotion",
                    options=['Semua'] + available_emotions,
                    index=0
                )
            else:
                selected_emotion_wc = 'Semua'
            
            # Platform specific wordcloud
            selected_platform_wc = st.sidebar.selectbox(
                "Pilih Platform untuk WordCloud",
                options=['Semua Platform'] + selected_platforms,
                index=0
            )
            
            # Filter data based on selections
            wc_data = filtered_sentiment.copy()
            
            if selected_sentiment != 'Semua':
                wc_data = wc_data[wc_data['sentiment'] == selected_sentiment]
            
            if selected_emotion_wc != 'Semua' and 'emotion' in wc_data.columns:
                wc_data = wc_data[wc_data['emotion'] == selected_emotion_wc]
            
            if selected_platform_wc != 'Semua Platform':
                wc_data = wc_data[wc_data['platform'] == selected_platform_wc]
            
            # Generate dynamic title
            title_parts = []
            if selected_platform_wc != 'Semua Platform':
                title_parts.append(f"Platform {selected_platform_wc}")
            if selected_sentiment != 'Semua':
                title_parts.append(f"Sentimen {selected_sentiment}")
            if selected_emotion_wc != 'Semua':
                title_parts.append(f"Emotion {selected_emotion_wc}")
            
            if title_parts:
                title = f"Word Cloud - {' | '.join(title_parts)}"
            else:
                title = "Word Cloud - Semua Data"
            
            st.subheader(title)
            
            if not wc_data.empty:
                fig = create_wordcloud(wc_data['content_text'], title)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada teks yang cukup untuk membuat word cloud dengan filter yang dipilih")
            else:
                st.info("Tidak ada data yang sesuai dengan filter yang dipilih")
            
            # Sentiment comparison wordclouds
            st.subheader("📊 Perbandingan WordCloud per Sentimen")
            
            if len(available_sentiments) > 1:
                cols = st.columns(len(available_sentiments))
                
                for i, sentiment in enumerate(available_sentiments):
                    sentiment_data = filtered_sentiment[filtered_sentiment['sentiment'] == sentiment]
                    
                    if selected_platform_wc != 'Semua Platform':
                        sentiment_data = sentiment_data[sentiment_data['platform'] == selected_platform_wc]
                    
                    if selected_emotion_wc != 'Semua' and 'emotion' in sentiment_data.columns:
                        sentiment_data = sentiment_data[sentiment_data['emotion'] == selected_emotion_wc]
                    
                    with cols[i]:
                        st.write(f"**{sentiment}**")
                        if not sentiment_data.empty:
                            fig = create_wordcloud(
                                sentiment_data['content_text'], 
                                f"WordCloud - {sentiment}"
                            )
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.info(f"Tidak ada teks untuk sentimen {sentiment}")
                        else:
                            st.info(f"Tidak ada data untuk sentimen {sentiment}")
            
            # Emotion comparison wordclouds
            if available_emotions and len(available_emotions) > 1:
                st.subheader("🎭 Perbandingan WordCloud per Emotion")
                
                cols = st.columns(min(3, len(available_emotions)))
                
                for i, emotion in enumerate(available_emotions[:6]):  # Limit to 6 emotions
                    emotion_data = filtered_sentiment[filtered_sentiment['emotion'] == emotion]
                    
                    if selected_platform_wc != 'Semua Platform':
                        emotion_data = emotion_data[emotion_data['platform'] == selected_platform_wc]
                    
                    if selected_sentiment != 'Semua':
                        emotion_data = emotion_data[emotion_data['sentiment'] == selected_sentiment]
                    
                    with cols[i % len(cols)]:
                        st.write(f"**{emotion}**")
                        if not emotion_data.empty:
                            fig = create_wordcloud(
                                emotion_data['content_text'], 
                                f"WordCloud - {emotion}"
                            )
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.info(f"Tidak ada teks untuk emotion {emotion}")
                        else:
                            st.info(f"Tidak ada data untuk emotion {emotion}")
            
            # Platform comparison with sentiment filter
            if len(selected_platforms) > 1:
                st.subheader("🏢 Perbandingan WordCloud per Platform")
                
                filter_info = []
                if selected_sentiment != 'Semua':
                    filter_info.append(f"Sentimen {selected_sentiment}")
                if selected_emotion_wc != 'Semua':
                    filter_info.append(f"Emotion {selected_emotion_wc}")
                
                if filter_info:
                    st.write(f"**Filter: {' | '.join(filter_info)}**")
                
                cols = st.columns(min(3, len(selected_platforms)))
                
                for i, platform in enumerate(selected_platforms):
                    platform_data = filtered_sentiment[filtered_sentiment['platform'] == platform]
                    
                    if selected_sentiment != 'Semua':
                        platform_data = platform_data[platform_data['sentiment'] == selected_sentiment]
                    
                    if selected_emotion_wc != 'Semua' and 'emotion' in platform_data.columns:
                        platform_data = platform_data[platform_data['emotion'] == selected_emotion_wc]
                    
                    with cols[i % len(cols)]:
                        st.write(f"**{platform}**")
                        if not platform_data.empty:
                            fig = create_wordcloud(
                                platform_data['content_text'],
                                f"WordCloud - {platform}"
                            )
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.info(f"Tidak ada teks untuk {platform}")
                        else:
                            st.info(f"Tidak ada data untuk {platform}")
    
    with tab6:
        st.header("🕸️ Social Network Analysis")
        
        if not filtered_sna.empty:
            # SNA Filter Controls
            st.subheader("🎛️ Filter Network Analysis")
            
            # Create emotion mapping from sentiment data for SNA
            emotion_mapping = {}
            if not filtered_sentiment.empty and 'emotion' in filtered_sentiment.columns:
                # Create mapping based on content similarity or user matching
                # For simplicity, we'll use a basic approach - in real scenario, you might want more sophisticated matching
                for _, row in filtered_sentiment.iterrows():
                    if pd.notna(row.get('content_text', '')) and pd.notna(row.get('emotion', '')):
                        # Use first few words as key for mapping
                        content_key = ' '.join(str(row['content_text']).split()[:5])
                        emotion_mapping[content_key] = row['emotion']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Platform filter for SNA
                sna_platforms = filtered_sna['platform'].unique().tolist()
                selected_sna_platforms = st.multiselect(
                    "Filter Platform SNA",
                    options=sna_platforms,
                    default=sna_platforms,
                    key="sna_platform_filter"
                )
            
            with col2:
                # Sentiment filter for SNA
                sna_sentiments = filtered_sna['sentiment'].unique().tolist() if 'sentiment' in filtered_sna.columns else []
                selected_sna_sentiments = st.multiselect(
                    "Filter Sentimen SNA",
                    options=sna_sentiments,
                    default=sna_sentiments,
                    key="sna_sentiment_filter"
                )
            
            with col3:
                # Emotion filter for SNA (from mapped data)
                if emotion_mapping:
                    available_emotions_sna = list(set(emotion_mapping.values()))
                    selected_sna_emotions = st.multiselect(
                        "Filter Emotion SNA",
                        options=available_emotions_sna,
                        default=available_emotions_sna,
                        key="sna_emotion_filter"
                    )
                else:
                    selected_sna_emotions = []
            
            with col4:
                # Issue/Topic filter for SNA
                if not filtered_sentiment.empty and 'content_text' in filtered_sentiment.columns:
                    # Extract topics for filter using KeyBERT
                    topics_for_filter = extract_topics(filtered_sentiment['content_text'], n_topics=20)
                    topic_options = ['Semua Isu'] + [topic for topic, _ in topics_for_filter]
                    
                    selected_issue = st.selectbox(
                        "Filter berdasarkan Isu (KeyBERT Topics)",
                        options=topic_options,
                        index=0,
                        key="sna_issue_filter",
                        help="Topics diekstrak menggunakan KeyBERT untuk analisis semantik yang lebih akurat"
                    )
                else:
                    selected_issue = 'Semua Isu'
            
            with col5:
                # Relation type filter
                relation_types = filtered_sna['relation'].unique().tolist() if 'relation' in filtered_sna.columns else []
                selected_relations = st.multiselect(
                    "Filter Jenis Relasi",
                    options=relation_types,
                    default=relation_types,
                    key="sna_relation_filter"
                )
            
            # Apply filters to SNA data
            sna_filtered = filtered_sna.copy()
            
            if selected_sna_platforms:
                sna_filtered = sna_filtered[sna_filtered['platform'].isin(selected_sna_platforms)]
            
            if selected_sna_sentiments and 'sentiment' in sna_filtered.columns:
                sna_filtered = sna_filtered[sna_filtered['sentiment'].isin(selected_sna_sentiments)]
            
            # Filter by emotion using mapping
            if selected_sna_emotions and emotion_mapping:
                emotion_filtered_indices = []
                for idx, row in sna_filtered.iterrows():
                    if pd.notna(row.get('content_text', '')):
                        content_key = ' '.join(str(row['content_text']).split()[:5])
                        if content_key in emotion_mapping and emotion_mapping[content_key] in selected_sna_emotions:
                            emotion_filtered_indices.append(idx)
                
                if emotion_filtered_indices:
                    sna_filtered = sna_filtered.loc[emotion_filtered_indices]
                else:
                    sna_filtered = sna_filtered.iloc[0:0]  # Empty dataframe
            
            if selected_relations and 'relation' in sna_filtered.columns:
                sna_filtered = sna_filtered[sna_filtered['relation'].isin(selected_relations)]
            
            # Filter by issue/topic
            if selected_issue != 'Semua Isu' and 'content_text' in sna_filtered.columns:
                sna_filtered = sna_filtered[
                    sna_filtered['content_text'].str.contains(
                        selected_issue, case=False, na=False
                    )
                ]
            
            # Display current filter status
            emotion_status = f"{len(selected_sna_emotions)} emotion" if selected_sna_emotions else "Semua emotion"
            st.info(f"📊 Filter Aktif: {len(selected_sna_platforms)} platform, "
                   f"{len(selected_sna_sentiments)} sentimen, "
                   f"{emotion_status}, "
                   f"{'Isu: ' + selected_issue if selected_issue != 'Semua Isu' else 'Semua isu'}, "
                   f"{len(selected_relations)} jenis relasi")
            
            if not sna_filtered.empty:
                # Network statistics
                st.subheader("📊 Statistik Jaringan")
                
                G, top_influencers, network_fig = create_network_graph(sna_filtered)
                
                if G:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Nodes", G.number_of_nodes())
                    
                    with col2:
                        st.metric("Total Edges", G.number_of_edges())
                    
                    with col3:
                        density = nx.density(G)
                        st.metric("Network Density", f"{density:.3f}")
                    
                    with col4:
                        st.metric("Data Points", len(sna_filtered))
                    
                    # Network Visualization
                    st.subheader("🕸️ Visualisasi Network")
                    
                    # Add description of current view
                    if selected_issue != 'Semua Isu':
                        st.write(f"**Fokus pada Isu: '{selected_issue}'**")
                    
                    if network_fig:
                        st.pyplot(network_fig)
                    else:
                        st.info("Tidak dapat membuat visualisasi network")
                    
                    # Network insights
                    st.subheader("🔍 Insight Network")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Degree distribution
                        degrees = [d for n, d in G.degree()]
                        avg_degree = np.mean(degrees)
                        max_degree = max(degrees)
                        
                        st.write("**Statistik Degree:**")
                        st.write(f"- Average Degree: {avg_degree:.2f}")
                        st.write(f"- Maximum Degree: {max_degree}")
                        st.write(f"- Total Connections: {sum(degrees)//2}")
                        
                        # Filter-specific insights
                        if selected_issue != 'Semua Isu':
                            st.write(f"- **Isu Focus**: {selected_issue}")
                        if len(selected_sna_platforms) < len(sna_platforms):
                            st.write(f"- **Platform**: {', '.join(selected_sna_platforms)}")
                    
                    with col2:
                        # Centrality measures
                        if G.number_of_nodes() <= 100:  # For performance
                            try:
                                betweenness = nx.betweenness_centrality(G)
                                top_betweenness = max(betweenness.items(), key=lambda x: x[1])
                                
                                closeness = nx.closeness_centrality(G)
                                top_closeness = max(closeness.items(), key=lambda x: x[1])
                                
                                st.write("**Top Centrality:**")
                                st.write(f"- Betweenness: {top_betweenness[0][:20]}...")
                                st.write(f"- Closeness: {top_closeness[0][:20]}...")
                            except:
                                st.write("**Centrality analysis tidak tersedia**")
                        else:
                            st.write("**Network terlalu besar untuk analisis centrality detail**")
                        
                        # Sentiment distribution in current network
                        if 'sentiment' in sna_filtered.columns and len(selected_sna_sentiments) > 1:
                            sentiment_dist = sna_filtered['sentiment'].value_counts()
                            st.write("**Distribusi Sentimen:**")
                            for sent, count in sentiment_dist.items():
                                pct = (count / len(sna_filtered)) * 100
                                st.write(f"- {sent}: {count} ({pct:.1f}%)")
                    
                    # Top influencers
                    st.subheader("🏆 Top Influencers")
                    
                    if top_influencers:
                        influencer_df = pd.DataFrame(top_influencers, columns=['User', 'Influence Score'])
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.bar(
                                influencer_df.head(10),
                                x='Influence Score',
                                y='User',
                                orientation='h',
                                title=f"Top 10 Influencers - {selected_issue if selected_issue != 'Semua Isu' else 'Semua Isu'}",
                                color='Influence Score',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write("**Top 10 Influencers:**")
                            for i, (user, score) in enumerate(top_influencers[:10], 1):
                                display_user = user[:20] + "..." if len(user) > 20 else user
                                st.write(f"{i}. {display_user} ({score})")
                    
                    # Relation types distribution
                    st.subheader("🔗 Distribusi Jenis Relasi")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        relation_counts = sna_filtered['relation'].value_counts()
                        fig = px.pie(
                            values=relation_counts.values,
                            names=relation_counts.index,
                            title=f"Distribusi Jenis Relasi - {selected_issue if selected_issue != 'Semua Isu' else 'Semua Isu'}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Jenis Relasi:**")
                        for relation, count in relation_counts.items():
                            percentage = (count / relation_counts.sum()) * 100
                            st.write(f"- {relation}: {count} ({percentage:.1f}%)")
                    
                    # Platform-specific network analysis
                    if len(selected_sna_platforms) > 1:
                        st.subheader("📱 Analisis Network per Platform")
                        
                        platform_stats = []
                        for platform in selected_sna_platforms:
                            platform_sna = sna_filtered[sna_filtered['platform'] == platform]
                            if not platform_sna.empty:
                                platform_G, _, _ = create_network_graph(platform_sna)
                                if platform_G:
                                    platform_stats.append({
                                        'Platform': platform,
                                        'Nodes': platform_G.number_of_nodes(),
                                        'Edges': platform_G.number_of_edges(),
                                        'Density': nx.density(platform_G),
                                        'Data Points': len(platform_sna)
                                    })
                        
                        if platform_stats:
                            platform_df = pd.DataFrame(platform_stats)
                            st.dataframe(platform_df, use_container_width=True)
                    
                    # Issue-specific analysis
                    if selected_issue != 'Semua Isu':
                        st.subheader(f"🎯 Analisis Spesifik Isu: '{selected_issue}'")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Karakteristik Network untuk Isu ini:**")
                            issue_connections = G.number_of_edges()
                            issue_participants = G.number_of_nodes()
                            
                            st.write(f"- Partisipan dalam diskusi: {issue_participants}")
                            st.write(f"- Total interaksi: {issue_connections}")
                            
                            if issue_connections > 0:
                                connectivity = issue_connections / max(1, (issue_participants * (issue_participants - 1) / 2))
                                st.write(f"- Tingkat keterhubungan: {connectivity:.3f}")
                        
                        with col2:
                            # Platform participation for this issue
                            if len(selected_sna_platforms) > 1:
                                platform_participation = sna_filtered['platform'].value_counts()
                                st.write("**Partisipasi per Platform:**")
                                for platform, count in platform_participation.items():
                                    pct = (count / len(sna_filtered)) * 100
                                    st.write(f"- {platform}: {count} ({pct:.1f}%)")
                
                else:
                    st.info("Tidak ada data network yang cukup untuk analisis dengan filter yang dipilih")
            else:
                st.warning("Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah pengaturan filter.")
    
    with tab7:
        st.header("🔥 Trending Topics")
        
        # Add info about KeyBERT
        with st.expander("ℹ️ Tentang KeyBERT Topic Extraction", expanded=False):
            st.write("""
            **🤖 KeyBERT Topic Extraction**
            
            Dashboard ini menggunakan teknologi KeyBERT (Keyword BERT) untuk mengekstrak topics yang lebih meaningful:
            
            - **Semantic Analysis**: Menggunakan model transformer multilingual untuk memahami makna semantik
            - **Phrase Detection**: Dapat mendeteksi frasa 1-3 kata yang bermakna, bukan hanya kata tunggal
            - **Relevance Score**: Setiap topic diberi skor berdasarkan relevansinya dengan konten
            - **Diversity**: Algorithm memastikan topics yang beragam, tidak hanya variasi kata yang sama
            - **Language Support**: Mendukung teks bahasa Indonesia dan bahasa campuran
            
            **Keunggulan vs Frequency-based:**
            - Lebih akurat dalam menangkap topik yang bermakna
            - Tidak terpengaruh kata-kata umum yang sering muncul tapi tidak bermakna
            - Mampu menangkap konteks dan hubungan antar kata
            """)
        
        st.success("✅ KeyBERT siap digunakan! Topics akan diekstrak secara otomatis.")
        
        # Top topics from sentiment data
        if not filtered_sentiment.empty and 'content_text' in filtered_sentiment.columns:
            st.subheader("🎯 Top Issues/Topics yang Diperbincangkan")
            
            # Show progress while extracting topics
            with st.spinner("🤖 Mengekstrak topics menggunakan KeyBERT... Mohon tunggu sebentar."):
                topics = extract_topics(filtered_sentiment['content_text'], n_topics=15)
            
            if topics:
                topics_df = pd.DataFrame(topics, columns=['Topic', 'Relevance Score'])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        topics_df.head(10),
                        x='Relevance Score',
                        y='Topic',
                        orientation='h',
                        title="Top 10 Trending Topics (berdasarkan KeyBERT Relevance Score)",
                        color='Relevance Score',
                        color_continuous_scale='plasma',
                        labels={'Relevance Score': 'KeyBERT Relevance Score'}
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        xaxis_title="KeyBERT Relevance Score"
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{y}</b><br>Relevance Score: %{x}<extra></extra>'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Klik untuk Analisis Detail:**")
                    selected_topic = st.selectbox(
                        "Pilih Topic untuk Analisis",
                        options=[None] + [topic for topic, _ in topics[:15]],
                        index=0,
                        key="topic_selector"
                    )
                
                # Detailed analysis for selected topic
                if selected_topic:
                    st.subheader(f"📊 Analisis Detail: '{selected_topic}'")
                    
                    # Filter posts containing the selected topic
                    topic_posts = filtered_sentiment[
                        filtered_sentiment['content_text'].str.contains(
                            selected_topic, case=False, na=False
                        )
                    ]
                    
                    if not topic_posts.empty:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Posts", len(topic_posts))
                        
                        with col2:
                            if 'views' in topic_posts.columns:
                                total_views = topic_posts['views'].sum()
                                st.metric("Total Views", f"{total_views:,}")
                        
                        with col3:
                            platforms_mentioned = topic_posts['platform'].nunique()
                            st.metric("Platforms", platforms_mentioned)
                        
                        # Sentiment distribution for this topic
                        st.subheader("📈 Distribusi Sentimen & Emotion")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_counts = topic_posts['sentiment'].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title=f"Sentimen untuk '{selected_topic}'",
                                color=sentiment_counts.index,
                                color_discrete_map={
                                    "Positif": "blue",
                                    "Negatif": "red",
                                    "Netral": "gray"
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Emotion distribution for this topic
                            if 'emotion' in topic_posts.columns:
                                emotion_counts = topic_posts['emotion'].value_counts()
                                fig = px.pie(
                                    values=emotion_counts.values,
                                    names=emotion_counts.index,
                                    title=f"Emotion untuk '{selected_topic}'",
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Data emotion tidak tersedia")
                        
                        # Cross-analysis: Sentiment vs Platform and Emotion vs Platform
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sentiment by platform
                            sentiment_platform = topic_posts.groupby(['platform', 'sentiment']).size().reset_index(name='count')
                            if not sentiment_platform.empty:
                                fig = px.bar(
                                    sentiment_platform,
                                    x='platform',
                                    y='count',
                                    color='sentiment',
                                    title="Sentimen per Platform",
                                    barmode='stack',
                                    color_discrete_map={
                                        "Positif": "#4a90e2",
                                        "Negatif": "#e74c3c",
                                        "Netral": "#95a5a6"
                                    }

                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Emotion by platform
                            if 'emotion' in topic_posts.columns:
                                emotion_platform = topic_posts.groupby(['platform', 'emotion']).size().reset_index(name='count')
                                if not emotion_platform.empty:
                                    fig = px.bar(
                                        emotion_platform,
                                        x='platform',
                                        y='count',
                                        color='emotion',
                                        title=f"Emotion per Platform - '{selected_topic}'",
                                        barmode='stack',
                                        color_discrete_sequence=px.colors.qualitative.Set3
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Emotion vs Sentiment correlation for this topic
                        if 'emotion' in topic_posts.columns:
                            st.subheader("🔗 Korelasi Emotion vs Sentiment untuk Topic ini")
                            
                            emotion_sentiment_topic = pd.crosstab(
                                topic_posts['emotion'], 
                                topic_posts['sentiment'], 
                                normalize='index'
                            ) * 100
                            
                            if not emotion_sentiment_topic.empty:
                                fig = px.imshow(
                                    emotion_sentiment_topic.values,
                                    x=emotion_sentiment_topic.columns,
                                    y=emotion_sentiment_topic.index,
                                    title=f"Heatmap Emotion vs Sentiment untuk '{selected_topic}' (%)",
                                    color_continuous_scale='RdYlBu_r',
                                    text_auto='.1f'
                                )
                                fig.update_layout(
                                    xaxis_title="Sentiment",
                                    yaxis_title="Emotion"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Trend over time for this topic
                        st.subheader("📅 Tren Temporal")
                        
                        # Daily trend
                        topic_posts['date'] = topic_posts['timestamp'].dt.date
                        daily_trend = topic_posts.groupby('date').size().reset_index(name='mentions')
                        daily_trend['date'] = pd.to_datetime(daily_trend['date'])
                        
                        if len(daily_trend) > 1:
                            fig = px.line(
                                daily_trend,
                                x='date',
                                y='mentions',
                                title=f"Tren Harian Pembahasan '{selected_topic}'"
                            )
                            fig.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Sebutan")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Sentiment trend over time
                        if len(topic_posts) > 5:  # Only if enough data
                            sentiment_trend = topic_posts.groupby(['date', 'sentiment']).size().reset_index(name='count')
                            sentiment_trend['date'] = pd.to_datetime(sentiment_trend['date'])
                            
                            if not sentiment_trend.empty:
                                fig = px.line(
                                    sentiment_trend,
                                    x='date',
                                    y='count',
                                    color='sentiment',
                                    title=f"Tren Sentimen dari Waktu ke Waktu - '{selected_topic}'"
                                )
                                fig.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Posts")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Platform analysis
                        st.subheader("🏢 Analisis Platform")
                        
                        agg_dict = {
                            'content_text': 'count',
                            'views': 'sum' if 'views' in topic_posts.columns else 'count',
                            'likes': 'sum' if 'likes' in topic_posts.columns else 'count'
                        }
                        
                        # Add emotion stats if available
                        if 'emotion' in topic_posts.columns:
                            # Most common emotion per platform
                            platform_emotion = topic_posts.groupby('platform')['emotion'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A')
                            platform_emotion_counts = topic_posts.groupby('platform')['emotion'].nunique()
                            
                        platform_stats = topic_posts.groupby('platform').agg(agg_dict).reset_index()
                        platform_stats.columns = ['Platform', 'Posts', 'Total Views', 'Total Likes']
                        
                        # Add emotion data
                        if 'emotion' in topic_posts.columns:
                            platform_stats['Dominant Emotion'] = platform_stats['Platform'].map(platform_emotion)
                            platform_stats['Unique Emotions'] = platform_stats['Platform'].map(platform_emotion_counts)
                        
                        platform_stats = platform_stats.sort_values('Posts', ascending=False)
                        
                        st.dataframe(platform_stats, use_container_width=True)
                        
                        # Sample posts
                        st.subheader("📝 Sample Posts")
                        
                        # Show top posts by engagement
                        post_columns = ['platform', 'content_text', 'sentiment', 'timestamp']
                        if 'views' in topic_posts.columns:
                            post_columns.append('views')
                        if 'emotion' in topic_posts.columns:
                            post_columns.append('emotion')
                        
                        if 'views' in topic_posts.columns:
                            top_posts = topic_posts.nlargest(3, 'views')[post_columns]
                        else:
                            top_posts = topic_posts.head(3)[post_columns]
                        
                        for idx, post in top_posts.iterrows():
                            # Create title with sentiment and emotion
                            title_parts = [f"Post from {post['platform']}", f"{post['sentiment']} sentiment"]
                            if 'emotion' in post and pd.notna(post['emotion']):
                                title_parts.append(f"{post['emotion']} emotion")
                            
                            with st.expander(" - ".join(title_parts)):
                                st.write(f"**Content:** {post['content_text'][:200]}...")
                                st.write(f"**Date:** {post['timestamp']}")
                                st.write(f"**Sentiment:** {post['sentiment']}")
                                if 'emotion' in post and pd.notna(post['emotion']):
                                    st.write(f"**Emotion:** {post['emotion']}")
                                if 'views' in post:
                                    st.write(f"**Views:** {post['views']:,}")
                    
                    else:
                        st.info(f"Tidak ada posts yang ditemukan untuk topic '{selected_topic}'")
                
                # Show all topics table
                st.subheader("📋 Daftar Lengkap Trending Topics")
                
                # Add explanation for the relevance score
                st.write("**Keterangan**: Relevance Score menunjukkan seberapa relevan dan representatif suatu topic/keyword terhadap keseluruhan konten berdasarkan analisis semantik KeyBERT.")
                
                st.dataframe(topics_df, use_container_width=True)
            else:
                st.info("Tidak dapat mengekstrak topics dari data")
        
        # Trending by platform
        if len(selected_platforms) > 1:
            st.subheader("🏢 Trending Topics per Platform")
            
            platform_cols = st.columns(len(selected_platforms))
            
            for i, platform in enumerate(selected_platforms):
                platform_data = filtered_sentiment[filtered_sentiment['platform'] == platform]
                
                with platform_cols[i]:
                    st.write(f"**{platform}**")
                    
                    if not platform_data.empty and 'content_text' in platform_data.columns:
                        platform_topics = extract_topics(platform_data['content_text'], n_topics=5)
                        
                        if platform_topics:
                            platform_topics_df = pd.DataFrame(platform_topics, columns=['Topic', 'Relevance Score'])
                            st.dataframe(platform_topics_df, use_container_width=True, height=200)
                        else:
                            st.info(f"Tidak ada topics untuk {platform}")
                    else:
                        st.info(f"Tidak ada data untuk {platform}")

if __name__ == "__main__":
    main() 