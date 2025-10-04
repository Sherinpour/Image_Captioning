#!/usr/bin/env python3
"""
Load Dataset Script for EDA Project
Loads all entities_dataset_v2.json files from the data folder and provides basic statistics.
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
    Print basic information about the loaded dataset using pandas.
    
    Args:
        df (pd.DataFrame): Dataset DataFrame
        output_file (file): Optional file object to write output to
    """
    def print_and_write(text):
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    print_and_write("\n" + "="*50)
    print_and_write("DATASET INFORMATION (PANDAS)")
    print_and_write("="*50)
    
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
    
    if not entries:
        print_and_write("No entries to analyze.")
        return {}
    
    print_and_write("\n" + "="*50)
    print_and_write("DATA TYPE ANALYSIS")
    print_and_write("="*50)
    
    # Get column names from first entry
    column_names = list(entries[0].keys())
    data_types = {}
    
    print_and_write(f"Analyzing data types for {len(column_names)} columns...")
    print_and_write(f"Sample size: {min(1000, len(entries))} entries (for type inference)")
    
    # Analyze each column
    for col_name in column_names:
        print_and_write(f"\nColumn: '{col_name}'")
        
        # Sample data for type inference (use first 1000 entries or all if less)
        sample_size = min(1000, len(entries))
        sample_values = [entry.get(col_name) for entry in entries[:sample_size]]
        
        # Remove None values for analysis
        non_null_values = [val for val in sample_values if val is not None]
        null_count = len(sample_values) - len(non_null_values)
        
        if not non_null_values:
            print_and_write(f"  Type: No data (all null)")
            print_and_write(f"  Null values: {null_count}/{len(sample_values)}")
            data_types[col_name] = "null"
            continue
        
        # Determine the primary data type
        primary_type = type(non_null_values[0]).__name__
        
        # Check if all values have the same type
        all_same_type = all(type(val).__name__ == primary_type for val in non_null_values)
        
        # Get unique types in the column
        unique_types = list(set(type(val).__name__ for val in non_null_values))
        
        print_and_write(f"  Primary type: {primary_type}")
        print_and_write(f"  All same type: {all_same_type}")
        print_and_write(f"  Unique types found: {unique_types}")
        print_and_write(f"  Null values: {null_count}/{len(sample_values)} ({null_count/len(sample_values)*100:.1f}%)")
        
        # Additional analysis based on type
        if primary_type == 'str':
            # String analysis
            lengths = [len(str(val)) for val in non_null_values]
            print_and_write(f"  String length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
            
            # Check if it looks like a URL
            url_like = sum(1 for val in non_null_values if str(val).startswith(('http://', 'https://')))
            if url_like > 0:
                print_and_write(f"  URL-like strings: {url_like}/{len(non_null_values)} ({url_like/len(non_null_values)*100:.1f}%)")
            
            # Check if it looks like a UUID
            uuid_like = sum(1 for val in non_null_values if len(str(val)) == 36 and str(val).count('-') == 4)
            if uuid_like > 0:
                print_and_write(f"  UUID-like strings: {uuid_like}/{len(non_null_values)} ({uuid_like/len(non_null_values)*100:.1f}%)")
        
        elif primary_type == 'list':
            # List analysis
            list_lengths = [len(val) for val in non_null_values if isinstance(val, list)]
            if list_lengths:
                print_and_write(f"  List length - Min: {min(list_lengths)}, Max: {max(list_lengths)}, Avg: {sum(list_lengths)/len(list_lengths):.1f}")
            
            # Check if it's a list of dictionaries (like entities)
            dict_lists = sum(1 for val in non_null_values if isinstance(val, list) and val and isinstance(val[0], dict))
            if dict_lists > 0:
                print_and_write(f"  List of dictionaries: {dict_lists}/{len(non_null_values)} ({dict_lists/len(non_null_values)*100:.1f}%)")
                
                # Analyze dictionary structure in lists
                if dict_lists > 0:
                    sample_dict = next(val[0] for val in non_null_values if isinstance(val, list) and val and isinstance(val[0], dict))
                    dict_keys = list(sample_dict.keys())
                    print_and_write(f"  Dictionary keys in lists: {dict_keys}")
        
        elif primary_type == 'int':
            # Integer analysis
            print_and_write(f"  Integer range - Min: {min(non_null_values)}, Max: {max(non_null_values)}")
        
        elif primary_type == 'float':
            # Float analysis
            print_and_write(f"  Float range - Min: {min(non_null_values):.2f}, Max: {max(non_null_values):.2f}")
        
        # Show sample values
        sample_values_display = non_null_values[:3]
        print_and_write(f"  Sample values: {sample_values_display}")
        
        data_types[col_name] = {
            'primary_type': primary_type,
            'all_same_type': all_same_type,
            'unique_types': unique_types,
            'null_percentage': null_count/len(sample_values)*100
        }
    
    return data_types

def main():
    """Main function to load and analyze the dataset."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_folder = project_root / "data"
    output_folder = project_root / "output"
    
    # Ensure output folder exists
    output_folder.mkdir(exist_ok=True)
    
    print("Loading entities dataset...")
    print(f"Data folder: {data_folder}")
    print(f"Output folder: {output_folder}")
    
    try:
        # Load the dataset
        entries = load_dataset(data_folder)
        
        # Create output files
        row_column_file = output_folder / "row_column_info.txt"
        data_types_file = output_folder / "data_types.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save row and column information
        with open(row_column_file, 'w', encoding='utf-8') as output_file:
            # Write header to file
            output_file.write(f"Dataset Row and Column Information\n")
            output_file.write(f"Generated on: {timestamp}\n")
            output_file.write(f"Data folder: {data_folder}\n")
            output_file.write("="*60 + "\n\n")
            
            # Print information about the dataset (both to console and file)
            num_rows, column_names, _ = print_dataset_info(entries, output_file)
            
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
            data_types_output.write(f"Dataset Data Type Analysis\n")
            data_types_output.write(f"Generated on: {timestamp}\n")
            data_types_output.write(f"Data folder: {data_folder}\n")
            data_types_output.write("="*60 + "\n\n")
            
            # Analyze data types (both to console and file)
            data_types = analyze_data_types(entries, data_types_output)
            
            # Write summary to file
            data_types_output.write("\n" + "="*60 + "\n")
            data_types_output.write("DATA TYPE SUMMARY\n")
            data_types_output.write("="*60 + "\n")
            for col_name, type_info in data_types.items():
                if isinstance(type_info, dict):
                    data_types_output.write(f"{col_name}: {type_info['primary_type']} (consistent: {type_info['all_same_type']}, nulls: {type_info['null_percentage']:.1f}%)\n")
                else:
                    data_types_output.write(f"{col_name}: {type_info}\n")
            data_types_output.write(f"\nData type analysis completed successfully.\n")
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"✅ Total entries: {len(entries)}")
        print(f"✅ Total columns: {len(column_names)}")
        print(f"✅ Column names: {', '.join(column_names)}")
        print(f"✅ Data is accessible and ready for analysis")
        print(f"✅ Row/column info saved to: {row_column_file}")
        print(f"✅ Data types analysis saved to: {data_types_file}")
        
        return entries
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    dataset = main()
