import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

def load_and_clean_data(file_path='metadata.csv'):
    """
    Load and clean the CORD-19 metadata dataset
    
    Parameters:
    file_path (str): Path to the metadata.csv file
    
    Returns:
    tuple: Cleaned DataFrame and analysis results
    """
    print("Loading data...")
    # Load the data with low_memory=False to avoid mixed type warnings
    df = pd.read_csv(file_path, low_memory=False)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Basic exploration
    print("\nData types:")
    print(df.dtypes.head(10))  # Show first 10 columns
    
    print("\nMissing values in key columns:")
    missing_data = df[['title', 'abstract', 'journal', 'publish_time']].isnull().sum()
    print(missing_data)
    
    # Data cleaning
    print("\nCleaning data...")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Handle missing values - keep rows with at least title
    df_clean = df_clean.dropna(subset=['title'])
    
    # Extract year from publication date
    # Handle different date formats safely
    def extract_year(date_str):
        try:
            if pd.isna(date_str):
                return np.nan
            # Convert to string and extract first 4 digits (year)
            date_str = str(date_str)
            year_match = re.search(r'(\d{4})', date_str)
            if year_match:
                return int(year_match.group(1))
            return np.nan
        except:
            return np.nan
    
    df_clean['year'] = df_clean['publish_time'].apply(extract_year)
    
    # Remove rows where year extraction failed or is unrealistic
    df_clean = df_clean.dropna(subset=['year'])
    df_clean = df_clean[df_clean['year'] > 2000]  # Reasonable filter for COVID papers
    
    # Create abstract word count
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Clean journal names
    df_clean['journal'] = df_clean['journal'].fillna('Unknown')
    df_clean['journal'] = df_clean['journal'].str.strip()
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    
    return df_clean

def analyze_data(df_clean):
    """
    Perform analysis on the cleaned dataset
    
    Parameters:
    df_clean (DataFrame): Cleaned dataset
    
    Returns:
    tuple: Analysis results including counts and visualizations
    """
    print("Analyzing data...")
    
    # Publications by year
    year_counts = df_clean['year'].value_counts().sort_index()
    
    # Top journals (excluding 'Unknown')
    known_journals = df_clean[df_clean['journal'] != 'Unknown']
    top_journals = known_journals['journal'].value_counts().head(15)
    
    # Word frequency in titles
    all_titles = ' '.join(df_clean['title'].dropna().astype(str).str.lower())
    # Remove common stopwords and punctuation
    words = re.findall(r'\b[a-z]{3,15}\b', all_titles)
    word_counts = Counter(words)
    # Remove common stopwords
    stopwords = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
                'as', 'an', 'from', 'that', 'is', 'are', 'this', 'which', 'be', 'at'}
    filtered_words = {word: count for word, count in word_counts.items() 
                     if word not in stopwords}
    title_words = pd.Series(filtered_words).sort_values(ascending=False).head(20)
    
    # Abstract length distribution
    abstract_lengths = df_clean['abstract_word_count']
    
    # Source distribution
    source_counts = df_clean['source_x'].value_counts().head(10)
    
    return year_counts, top_journals, title_words, abstract_lengths, source_counts

def create_visualizations(year_counts, top_journals, title_words, abstract_lengths, source_counts):
    """
    Create and save visualizations
    
    Parameters:
    Various analysis results from analyze_data function
    """
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    
    # 1. Publications by year
    plt.figure(figsize=(12, 6))
    year_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of COVID-19 Publications by Year', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/publications_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top journals
    plt.figure(figsize=(12, 6))
    top_journals.plot(kind='bar', color='lightgreen')
    plt.title('Top 15 Journals by Number of COVID-19 Publications', fontsize=16)
    plt.xlabel('Journal', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('images/top_journals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word frequency in titles
    plt.figure(figsize=(12, 6))
    title_words.head(10).plot(kind='bar', color='lightcoral')
    plt.title('Top 10 Words in COVID-19 Paper Titles', fontsize=16)
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/title_words.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Abstract length distribution
    plt.figure(figsize=(12, 6))
    plt.hist(abstract_lengths[abstract_lengths < 1000], bins=50, color='orange', alpha=0.7)
    plt.title('Distribution of Abstract Word Counts', fontsize=16)
    plt.xlabel('Number of Words in Abstract', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/abstract_lengths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Word cloud of titles
    all_titles = ' '.join([str(title) for title in df_clean['title'].dropna()])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Paper Titles', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/title_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to images/ folder")

if __name__ == "__main__":
    # Run the analysis if this script is executed directly
    df_clean = load_and_clean_data('metadata.csv')
    analysis_results = analyze_data(df_clean)
    create_visualizations(*analysis_results)
    
    # Print some summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total papers: {len(df_clean):,}")
    print(f"Time range: {df_clean['year'].min()} - {df_clean['year'].max()}")
    print(f"Journals with most papers: {df_clean['journal'].value_counts().index[0]}")
    print(f"Average abstract length: {df_clean['abstract_word_count'].mean():.1f} words")
