import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import load_and_clean_data, analyze_data
import os

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .section-header {font-size: 2rem; color: #ff7f0e; margin-top: 2rem;}
    .info-text {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data to improve performance"""
    try:
        df_clean = load_and_clean_data('metadata.csv')
        year_counts, top_journals, title_words, abstract_lengths, source_counts = analyze_data(df_clean)
        return df_clean, year_counts, top_journals, title_words, abstract_lengths, source_counts
    except FileNotFoundError:
        st.error("Error: metadata.csv file not found. Please make sure it's in the same directory as this app.")
        st.stop()

def main():
    # Header
    st.markdown('<h1 class="main-header">CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.write("Interactive exploration of COVID-19 research papers metadata from the CORD-19 dataset")
    
    # Load data with caching
    with st.spinner('Loading data... This may take a few minutes for the first time.'):
        df_clean, year_counts, top_journals, title_words, abstract_lengths, source_counts = load_data()
    
    # Sidebar
    st.sidebar.header("Filters and Controls")
    
    # Year range filter
    min_year = int(year_counts.index.min())
    max_year = int(year_counts.index.max())
    year_range = st.sidebar.slider(
        "Select publication year range",
        min_year, max_year, (2020, max_year)
    )
    
    # Journal filter
    top_journal_list = top_journals.index.tolist()
    selected_journals = st.sidebar.multiselect(
        "Filter by journal (top 15 shown)",
        options=top_journal_list,
        default=top_journal_list[:3] if len(top_journal_list) > 3 else top_journal_list
    )
    
    # Abstract length filter
    min_abstract, max_abstract = st.sidebar.slider(
        "Abstract word count range",
        int(abstract_lengths.min()), 
        int(abstract_lengths.quantile(0.95)),  # Exclude extreme outliers
        (0, 300)
    )
    
    # Apply filters
    filtered_df = df_clean[
        (df_clean['year'] >= year_range[0]) & 
        (df_clean['year'] <= year_range[1]) &
        (df_clean['journal'].isin(selected_journals) if selected_journals else True) &
        (df_clean['abstract_word_count'] >= min_abstract) &
        (df_clean['abstract_word_count'] <= max_abstract)
    ]
    
    # Main content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers", f"{len(df_clean):,}")
    with col2:
        st.metric("Filtered Papers", f"{len(filtered_df):,}")
    with col3:
        st.metric("Average Abstract Length", f"{filtered_df['abstract_word_count'].mean():.1f} words")
    
    # Dataset overview
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Key Statistics:**")
        st.write(f"- Time range: {df_clean['year'].min()} - {df_clean['year'].max()}")
        st.write(f"- Number of unique journals: {df_clean['journal'].nunique()}")
        st.write(f"- Papers with abstracts: {df_clean['abstract'].notna().sum():,}")
        st.write(f"- Most common journal: {df_clean['journal'].value_counts().index[0]}")
    
    with col2:
        st.markdown("**Filter Information:**")
        st.write(f"- Selected years: {year_range[0]} - {year_range[1]}")
        st.write(f"- Selected journals: {len(selected_journals)} of {len(top_journal_list)}")
        st.write(f"- Abstract length: {min_abstract} - {max_abstract} words")
    
    # Visualizations
    st.markdown('<h2 class="section-header">Visualizations</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Publications Over Time", 
        "Journal Analysis", 
        "Content Analysis", 
        "Sample Data"
    ])
    
    with tab1:
        st.subheader("Publications by Year")
        fig, ax = plt.subplots(figsize=(10, 6))
        year_counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Number of COVID-19 Publications by Year', fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Publications', fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Add some insights
        st.markdown("""
        **Insights:**
        - The number of COVID-19 publications exploded in 2020
        - Research output remained high in subsequent years
        - This reflects the global scientific response to the pandemic
        """)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Journals")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_journals.head(10).plot(kind='bar', color='lightgreen', ax=ax)
            ax.set_title('Top 10 Journals by Publications', fontsize=16)
            ax.set_xlabel('Journal', fontsize=12)
            ax.set_ylabel('Number of Publications', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Abstract Length Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(abstract_lengths[abstract_lengths < 1000], bins=50, 
                   color='orange', alpha=0.7)
            ax.set_title('Distribution of Abstract Word Counts', fontsize=16)
            ax.set_xlabel('Number of Words in Abstract', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Common Words in Titles")
            fig, ax = plt.subplots(figsize=(10, 6))
            title_words.head(10).plot(kind='bar', color='lightcoral', ax=ax)
            ax.set_title('Top 10 Words in Paper Titles', fontsize=16)
            ax.set_xlabel('Word', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Data Sources")
            fig, ax = plt.subplots(figsize=(10, 6))
            source_counts.plot(kind='bar', color='purple', alpha=0.7, ax=ax)
            ax.set_title('Top Data Sources', fontsize=16)
            ax.set_xlabel('Source', fontsize=12)
            ax.set_ylabel('Number of Papers', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Sample of Filtered Data")
        st.dataframe(
            filtered_df[['title', 'journal', 'year', 'abstract_word_count']].head(20),
            height=400,
            use_container_width=True
        )
        
        # Download button for filtered data
        csv = filtered_df[['title', 'journal', 'year', 'abstract_word_count']].to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_cord19_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** 
    This application explores the CORD-19 dataset, which contains metadata about COVID-19 research papers.
    The data is from the [CORD-19 dataset on Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).
    """)

if __name__ == "__main__":
    main()
