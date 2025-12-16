import pandas as pd
import os
import re

INPUT_FILE = r"c:\Users\safal\Desktop\shl_assignment\data\raw\shl_catalogue.csv"
OUTPUT_DIR = r"c:\Users\safal\Desktop\shl_assignment\data\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shl_catalogue_clean.csv")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove "Description" prefix if present (case insensitive)
    text = re.sub(r'^Description\s*', '', text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    initial_count = len(df)
    print(f"Initial row count: {initial_count}")

    # 1. Deduplicate
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=['assessment_url'], keep='first')
    print(f"Rows after deduplication: {len(df)}")

    # 2. Filter empty URLs or Names
    print("Filtering invalid rows...")
    df = df.dropna(subset=['assessment_url', 'assessment_name'])
    df = df[df['assessment_url'].str.strip() != '']
    print(f"Rows after filtering: {len(df)}")

    # 3. Clean Description
    print("Cleaning descriptions...")
    df['description'] = df['description'].fillna("")
    df['description'] = df['description'].apply(clean_text)
    
    # Drops rows where description is empty after cleaning
    initial_len = len(df)
    df = df[df['description'] != ""]
    print(f"Dropped {initial_len - len(df)} rows with empty descriptions.")
    print(f"Rows after description filtering: {len(df)}")

    # 4. Normalize Test Type
    print("Normalizing test types...")
    def normalize_type(t):
        t = str(t).strip().upper()
        # Mapping rules based on analysis
        if t in ['K', 'S', 'A', 'AS']: # Knowledge, Simulation, Ability
            return 'K'
        elif t in ['P', 'B', 'D', 'C', 'BS', 'SB', 'E']: # Personality, Behavioral, Dev, Competency
            return 'P'
        else:
            return 'K' # Default fallback if unknown, or typically K

    df['test_type'] = df['test_type'].apply(normalize_type)
    print(f"Test type distribution:\n{df['test_type'].value_counts()}")
    
    # 5. Sanity Checks
    print("Running sanity checks...")
    current_count = len(df)
    if current_count < 377:
        print(f"WARNING: Row count {current_count} is less than required 377!")
    else:
        print(f"Row count check passed: {current_count} >= 377")

    missing_urls = df['assessment_url'].isnull().sum()
    if missing_urls > 0:
        print(f"WARNING: Found {missing_urls} rows with missing URLs!")

    # 6. Save
    # print("Dropping assessment_url column...")
    # df = df.drop(columns=['assessment_url'], errors='ignore')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
