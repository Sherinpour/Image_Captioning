#!/usr/bin/env python3
"""
Load Dataset Script for EDA Project (Pandas Version)
Loads all entities_dataset_v2.json files from the data folder and provides comprehensive EDA using pandas.
"""

import json
import os
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a backend that works in headless environments
import matplotlib
matplotlib.use('Agg')

def load_dataset(data_folder):
    """
    Load all JSON files from the entities_dataset_v2 directory and convert to pandas DataFrame.
    
    Args:
        data_folder (str): Path to the data folder containing entities_dataset_v2
        
    Returns:
        pd.DataFrame: Combined DataFrame of all entries from all JSON files
    """
    dataset_path = Path(data_folder) / "entities_dataset_v2"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    all_entries = []
    json_files = list(dataset_path.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for i, json_file in enumerate(json_files, 1):
        try:
            print(f"Loading {json_file.name} ({i}/{len(json_files)})...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_entries.extend(data)
                else:
                    print(f"Warning: {json_file.name} does not contain a list")
        except json.JSONDecodeError as e:
            print(f"Error loading {json_file.name}: {e}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # Convert to pandas DataFrame
    print("Converting to pandas DataFrame...")
    df = pd.DataFrame(all_entries)
    
    return df

def print_dataset_info(df, output_file=None):
    """
    Print comprehensive information about the loaded dataset using pandas.
    
    Args:
        df (pd.DataFrame): Dataset DataFrame
        output_file (file): Optional file object to write output to
    """
    def print_and_write(text):
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    print_and_write("\n" + "="*60)
    print_and_write("DATASET INFORMATION (PANDAS)")
    print_and_write("="*60)
    
    # Basic DataFrame info
    num_rows, num_cols = df.shape
    print_and_write(f"Dataset shape: {num_rows} rows × {num_cols} columns")
    print_and_write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print_and_write(f"\nColumn Information:")
    print_and_write(f"Columns: {list(df.columns)}")
    print_and_write(f"Data types:\n{df.dtypes}")
    
    # Missing values
    print_and_write(f"\nMissing Values:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print_and_write(missing_df.to_string())
    
    # Basic statistics for numeric columns
    print_and_write(f"\nNumeric Columns Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print_and_write(df[numeric_cols].describe().to_string())
    else:
        print_and_write("No numeric columns found.")
    
    # String columns analysis
    print_and_write(f"\nString Columns Analysis:")
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        print_and_write(f"\n{col}:")
        print_and_write(f"  - Unique values: {df[col].nunique()}")
        print_and_write(f"  - Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
        print_and_write(f"  - Sample values: {df[col].head(3).tolist()}")
    
    # Entities analysis
    if 'entities' in df.columns:
        print_and_write(f"\nEntities Analysis:")
        try:
            entity_counts = df['entities'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            print_and_write(f"  - Average entities per entry: {entity_counts.mean():.2f}")
            print_and_write(f"  - Min entities per entry: {entity_counts.min()}")
            print_and_write(f"  - Max entities per entry: {entity_counts.max()}")
            
            # Most common entity names
            all_entity_names = []
            for entities in df['entities']:
                if isinstance(entities, list):
                    for entity in entities:
                        if isinstance(entity, dict) and 'name' in entity:
                            all_entity_names.append(entity['name'])
            
            if all_entity_names:
                entity_name_counts = Counter(all_entity_names)
                print_and_write(f"  - Most common entity types:")
                for name, count in entity_name_counts.most_common(5):
                    print_and_write(f"    {name}: {count}")
        except Exception as e:
            print_and_write(f"  - Error analyzing entities: {e}")
            print_and_write(f"  - Entities column contains complex data structures")
    
    return num_rows, list(df.columns)

def analyze_data_types(df, output_file=None):
    """
    Analyze and print data types of each column in the dataset using pandas.
    
    Args:
        df (pd.DataFrame): Dataset DataFrame
        output_file (file): Optional file object to write output to
    """
    def print_and_write(text):
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    if df.empty:
        print_and_write("No data to analyze.")
        return {}
    
    print_and_write("\n" + "="*60)
    print_and_write("DATA TYPE ANALYSIS (PANDAS)")
    print_and_write("="*60)
    
    # Basic info
    print_and_write(f"Dataset shape: {df.shape}")
    print_and_write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types summary
    print_and_write(f"\nData Types Summary:")
    dtype_counts = df.dtypes.value_counts()
    print_and_write(dtype_counts.to_string())
    
    # Detailed analysis for each column
    print_and_write(f"\nDetailed Column Analysis:")
    data_types = {}
    
    for col in df.columns:
        print_and_write(f"\nColumn: '{col}'")
        
        # Basic info
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_percent = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        print_and_write(f"  Data type: {dtype}")
        print_and_write(f"  Non-null values: {len(df) - null_count}/{len(df)} ({100-null_percent:.1f}%)")
        print_and_write(f"  Unique values: {unique_count}")
        
        # Type-specific analysis
        if df[col].dtype == 'object':
            # String analysis
            non_null_series = df[col].dropna()
            if not non_null_series.empty:
                lengths = non_null_series.str.len()
                print_and_write(f"  String length - Min: {lengths.min()}, Max: {lengths.max()}, Avg: {lengths.mean():.1f}")
                
                # Check for URL patterns
                url_pattern = non_null_series.str.contains(r'^https?://', na=False)
                url_count = url_pattern.sum()
                if url_count > 0:
                    print_and_write(f"  URL-like strings: {url_count}/{len(non_null_series)} ({url_count/len(non_null_series)*100:.1f}%)")
                
                # Check for UUID patterns
                uuid_pattern = non_null_series.str.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', na=False)
                uuid_count = uuid_pattern.sum()
                if uuid_count > 0:
                    print_and_write(f"  UUID-like strings: {uuid_count}/{len(non_null_series)} ({uuid_count/len(non_null_series)*100:.1f}%)")
        
        elif 'int' in dtype or 'float' in dtype:
            # Numeric analysis
            print_and_write(f"  Range - Min: {df[col].min()}, Max: {df[col].max()}")
            print_and_write(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
        
        # Sample values
        sample_values = df[col].dropna().head(3).tolist()
        print_and_write(f"  Sample values: {sample_values}")
        
        data_types[col] = {
            'dtype': dtype,
            'null_percentage': null_percent,
            'unique_count': unique_count
        }
    
    return data_types

def create_visualizations(df, output_folder):
    """
    Create basic visualizations for the dataset.
    
    Args:
        df (pd.DataFrame): Dataset DataFrame
        output_folder (Path): Path to output folder
    """
    print("\nCreating visualizations...")
    
    try:
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Entity count distribution
        if 'entities' in df.columns:
            try:
                entity_counts = df['entities'].apply(lambda x: len(x) if isinstance(x, list) else 0)
                
                plt.figure(figsize=(10, 6))
                plt.hist(entity_counts, bins=20, edgecolor='black', alpha=0.7)
                plt.title('Distribution of Entity Counts per Entry')
                plt.xlabel('Number of Entities')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_folder / 'entity_count_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create entity count visualization: {e}")
        
        # 2. Group distribution
        if 'group' in df.columns:
            try:
                group_counts = df['group'].value_counts().head(10)
                
                plt.figure(figsize=(12, 8))
                group_counts.plot(kind='bar')
                plt.title('Top 10 Product Groups')
                plt.xlabel('Group')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_folder / 'group_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create group distribution visualization: {e}")
        
        # 3. Product distribution
        if 'product' in df.columns:
            try:
                product_counts = df['product'].value_counts().head(15)
                
                plt.figure(figsize=(12, 8))
                product_counts.plot(kind='bar')
                plt.title('Top 15 Product Types')
                plt.xlabel('Product Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_folder / 'product_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create product distribution visualization: {e}")
        
        # 4. String length distributions
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col not in ['entities']:  # Skip entities column
                try:
                    lengths = df[col].str.len()
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
                    plt.title(f'Distribution of {col} String Lengths')
                    plt.xlabel('String Length')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_folder / f'{col}_length_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not create {col} length distribution visualization: {e}")
        
        print("Visualizations saved to output folder.")
        
    except Exception as e:
        print(f"Warning: Error in visualization creation: {e}")
        print("Continuing without visualizations...")

def main():
    """Main function to load and analyze the dataset."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_folder = project_root / "data"
    output_folder = project_root / "output"
    
    # Ensure output folder exists
    output_folder.mkdir(exist_ok=True)
    
    print("Loading entities dataset with pandas...")
    print(f"Data folder: {data_folder}")
    print(f"Output folder: {output_folder}")
    
    try:
        # Load the dataset
        df = load_dataset(data_folder)
        
        # Create output files
        row_column_file = output_folder / "row_column_info_pandas.txt"
        data_types_file = output_folder / "data_types_pandas.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save row and column information
        with open(row_column_file, 'w', encoding='utf-8') as output_file:
            # Write header to file
            output_file.write(f"Dataset Row and Column Information (Pandas)\n")
            output_file.write(f"Generated on: {timestamp}\n")
            output_file.write(f"Data folder: {data_folder}\n")
            output_file.write("="*60 + "\n\n")
            
            # Print information about the dataset (both to console and file)
            num_rows, column_names = print_dataset_info(df, output_file)
            
            # Write summary to file
            output_file.write("\n" + "="*60 + "\n")
            output_file.write("SUMMARY\n")
            output_file.write("="*60 + "\n")
            output_file.write(f"Total Rows (Entries): {num_rows}\n")
            output_file.write(f"Total Columns: {len(column_names)}\n")
            output_file.write(f"Column Names: {', '.join(column_names)}\n")
            output_file.write(f"\nDataset loaded successfully and ready for analysis.\n")
        
        # Save data type analysis
        with open(data_types_file, 'w', encoding='utf-8') as data_types_output:
            # Write header to file
            data_types_output.write(f"Dataset Data Type Analysis (Pandas)\n")
            data_types_output.write(f"Generated on: {timestamp}\n")
            data_types_output.write(f"Data folder: {data_folder}\n")
            data_types_output.write("="*60 + "\n\n")
            
            # Analyze data types (both to console and file)
            data_types = analyze_data_types(df, data_types_output)
            
            # Write summary to file
            data_types_output.write("\n" + "="*60 + "\n")
            data_types_output.write("DATA TYPE SUMMARY\n")
            data_types_output.write("="*60 + "\n")
            for col_name, type_info in data_types.items():
                if isinstance(type_info, dict):
                    data_types_output.write(f"{col_name}: {type_info['dtype']} (nulls: {type_info['null_percentage']:.1f}%, unique: {type_info['unique_count']})\n")
                else:
                    data_types_output.write(f"{col_name}: {type_info}\n")
            data_types_output.write(f"\nData type analysis completed successfully.\n")
        
        # Create visualizations
        try:
            create_visualizations(df, output_folder)
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
            print("Continuing without visualizations...")
        
        print(f"\n✅ Dataset loaded successfully with pandas!")
        print(f"✅ Total entries: {len(df)}")
        print(f"✅ Total columns: {len(column_names)}")
        print(f"✅ Column names: {', '.join(column_names)}")
        print(f"✅ Data is accessible and ready for analysis")
        print(f"✅ Row/column info saved to: {row_column_file}")
        print(f"✅ Data types analysis saved to: {data_types_file}")
        print(f"✅ Visualizations saved to: {output_folder}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    dataset = main()
