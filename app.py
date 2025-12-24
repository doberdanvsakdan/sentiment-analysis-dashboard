import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from datetime import date
import calendar
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Advanced Sentiment Dashboard",
    page_icon="⭐",
    layout="wide"
)

# ==========================================
# 2. DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_data(file_name, page_type=None):
    try:
        df = pd.read_csv(file_name)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors="coerce")
        
        if page_type == "products" and 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['price'] = df['price'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")

        df.index = range(1, len(df) + 1)
        return df
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_sentiment_analyzer():
    # Use a much smaller model to stay under Render's 512MB limit
    return pipeline("sentiment-analysis", model="pysentimiento/bertweet-ca-sentiment") 

analyzer = get_sentiment_analyzer()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def generate_wordcloud(text, title, colormap):
    if not text or not text.strip(): return None
    wc = WordCloud(width=800, height=400, background_color="white", colormap=colormap, max_words=50).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, fontsize=20)
    return fig

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews Analysis"])

# ==========================================
# 5. PAGE: PRODUCTS & TESTIMONIALS
# ==========================================
if page == "Products":
    st.title("📦 Product Inventory")
    df_p = load_data("products_data.csv", page_type="products")
    if not df_p.empty:
        st.dataframe(df_p.drop(columns=['price_raw'], errors='ignore'), use_container_width=True)

elif page == "Testimonials":
    st.title("💬 Customer Testimonials")
    df_t = load_data("testimonials_data.csv")
    if not df_t.empty:
        clean_t = df_t.drop(columns=['date_raw', 'date'], errors='ignore')
        st.table(clean_t)

# ==========================================
# 6. PAGE: REVIEWS ANALYSIS (FIXED SYNC)
# ==========================================
elif page == "Reviews Analysis":
    st.title("⭐ Reviews Sentiment & Trend Analysis")
    df_r = load_data("reviews_data.csv")
    
    if not df_r.empty:
        df_r = df_r.dropna(subset=['date']).sort_values('date')
        df_r['year'] = df_r['date'].dt.year
        available_years = sorted(df_r['year'].unique())
        
        # Determine strict data bounds
        abs_min_date = df_r['date'].min().replace(day=1).date()
        abs_max_date = df_r['date'].max().replace(day=1).date()

        # --- SYNC LOGIC (WITH BOUNDARY CLIPPING) ---
        if 'date_range' not in st.session_state:
            st.session_state.date_range = (abs_min_date, abs_max_date)

        def sync_from_year():
            choice = st.session_state.year_radio
            if choice == "All":
                st.session_state.date_range = (abs_min_date, abs_max_date)
            else:
                y = int(choice)
                # Clip the start/end so they never go below abs_min or above abs_max
                new_start = max(abs_min_date, date(y, 1, 1))
                new_end = min(abs_max_date, date(y, 12, 1))
                st.session_state.date_range = (new_start, new_end)

        # --- FILTER UI ---
        st.subheader("📅 Temporal Filters")
        
        year_options = ["All"] + [str(y) for y in available_years]
        st.radio("Quick Select Year:", year_options, index=0, 
                 key="year_radio", on_change=sync_from_year, horizontal=True)

        # The Slider (Safely constrained to abs_min and abs_max)
        st.slider(
            "Select Month Range",
            min_value=abs_min_date,
            max_value=abs_max_date,
            key="date_range",
            format="MMM YYYY"
        )

        # Filter Masking
        start_m, end_m = st.session_state.date_range
        last_day = calendar.monthrange(end_m.year, end_m.month)[1]
        mask = (df_r['date'].dt.date >= start_m) & (df_r['date'].dt.date <= date(end_m.year, end_m.month, last_day))
        filtered_df = df_r.loc[mask].copy()

        st.info(f"Analysis Period: **{start_m.strftime('%B %Y')}** to **{end_m.strftime('%B %Y')}** ({len(filtered_df)} reviews)")

        # --- VISUALS ---
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Reviews Selected", len(filtered_df))
        m2.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f} ⭐")

        if st.button("🚀 Run AI Analysis"):
            with st.spinner("Calculating..."):
                texts = filtered_df['review'].fillna("").astype(str).tolist()
                results = analyzer(texts, truncation=True)
                filtered_df['Sentiment'] = [r['label'] for r in results]
                filtered_df['Confidence'] = [round(r['score'], 4) for r in results]

                # Trend Line
                trend_df = filtered_df.copy()
                trend_df['Month'] = trend_df['date'].dt.to_period('M').astype(str)
                trend_data = trend_df.groupby(['Month', 'Sentiment']).size().reset_index(name='Count')
                st.plotly_chart(px.line(trend_data, x='Month', y='Count', color='Sentiment', 
                                       color_discrete_map={'POSITIVE':'#00CC96','NEGATIVE':'#EF553B'}, markers=True), use_container_width=True)

                # Word Clouds
                wc1, wc2 = st.columns(2)
                pos_txt = " ".join(filtered_df[filtered_df['Sentiment'] == 'POSITIVE']['review'].astype(str))
                neg_txt = " ".join(filtered_df[filtered_df['Sentiment'] == 'NEGATIVE']['review'].astype(str))
                with wc1:
                    f1 = generate_wordcloud(pos_txt, "Positive Themes", "Greens")
                    if f1: st.pyplot(f1)
                with wc2:
                    f2 = generate_wordcloud(neg_txt, "Negative Themes", "Reds")
                    if f2: st.pyplot(f2)

                # Confidence Metrics
                avg_conf = filtered_df.groupby('Sentiment')['Confidence'].mean()
                c1, c2 = st.columns(2)
                if "POSITIVE" in avg_conf: c1.metric("Pos Confidence", f"{avg_conf['POSITIVE']:.2%}")
                if "NEGATIVE" in avg_conf: c2.metric("Neg Confidence", f"{avg_conf['NEGATIVE']:.2%}")

                # Final Table
                final_view = filtered_df.copy()
                final_view['date'] = final_view['date'].dt.strftime('%d/%m/%Y')
                st.dataframe(final_view[['date', 'review', 'rating', 'Sentiment', 'Confidence']], use_container_width=True)
        else:
            prev_view = filtered_df.copy()
            prev_view['date'] = prev_view['date'].dt.strftime('%d/%m/%Y')
            st.dataframe(prev_view[['date', 'review', 'rating']], use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("v6.1 - Safety Fixed Edition")
