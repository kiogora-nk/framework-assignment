import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data(num_rows=100):
    """Create a sample COVID-19 metadata dataset"""
    
    # Sample data components
    journals = [
        "Journal of Medical Research", "The Lancet", "New England Journal of Medicine",
        "Nature", "Science", "BMJ", "JAMA", "PLOS One", "PubMed", "Clinical Medicine",
        "Epidemiology", "Public Health", "Virology", "Immunology", "Unknown"
    ]
    
    sources = ["PMC", "WHO", "CZI", "NIH", "Other"]
    
    covid_words = [
        "COVID", "SARS", "Coronavirus", "Pandemic", "Vaccine", "Transmission",
        "Lockdown", "Mask", "Distance", "Infection", "Variant", "Immunity",
        "Treatment", "Symptoms", "Testing", "Spread", "Outbreak", "Isolation"
    ]
    
    common_words = [
        "Study", "Analysis", "Review", "Case", "Report", "Clinical", "Trial",
        "Effect", "Impact", "Model", "Data", "Results", "Findings", "Patients",
        "Health", "Public", "Medical", "Research", "Evaluation", "Assessment"
    ]
    
    # Generate data
    data = []
    for i in range(num_rows):
        # Generate title
        title_words = np.random.choice(covid_words, size=3, replace=False).tolist() + \
                     np.random.choice(common_words, size=2, replace=False).tolist()
        np.random.shuffle(title_words)
        title = " ".join(title_words) + ": A Comprehensive Analysis"
        
        # Generate abstract
        abstract = f"This study examines {np.random.choice(covid_words)} in the context of {np.random.choice(common_words)}. " \
                  f"Our findings suggest significant implications for {np.random.choice(['public health', 'clinical practice', 'policy making'])}. " \
                  f"Further research is needed to confirm these results."
        
        # Generate dates between 2020-2022
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2022, 12, 31)
        random_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
        
        data.append({
            'cord_uid': f'sample{i:06d}',
            'title': title,
            'abstract': abstract,
            'journal': np.random.choice(journals, p=[0.1, 0.08, 0.07, 0.06, 0.05, 0.09, 0.08, 0.12, 0.1, 0.05, 0.04, 0.03, 0.02, 0.02, 0.09]),
            'publish_time': random_date.strftime('%Y-%m-%d'),
            'authors': f"Researcher{i%10}, Coauthor{(i+1)%10}, Collaborator{(i+2)%10}",
            'url': f"https://example.com/paper{i}",
            'source_x': np.random.choice(sources)
        })
    
    return pd.DataFrame(data)

# Create and save sample data
if __name__ == "__main__":
    print("Creating sample metadata...")
    sample_df = create_sample_data(200)  # Create 200 sample rows
    sample_df.to_csv('sample_metadata.csv', index=False)
    print(f"Sample data saved to sample_metadata.csv with {len(sample_df)} rows")
    print(sample_df.head())
