#!/usr/bin/env python3
"""
EDA Charts Creation Script
Creates comprehensive visualizations for the entities dataset using matplotlib and seaborn.
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

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_dataset(data_folder):
    """Load dataset and convert to pandas DataFrame."""
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
            if i % 50 == 0:  # Show progress every 50 files
                print(f"Loading {i}/{len(json_files)} files...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_entries.extend(data)
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print("Converting to pandas DataFrame...")
    df = pd.DataFrame(all_entries)
    return df

def create_basic_overview_charts(df, output_folder):
    """Create basic overview charts."""
    print("Creating basic overview charts...")
    
    # 1. Dataset Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
    
    # Missing values heatmap
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    
    sns.heatmap(missing_df[['Missing Percentage']].T, 
                annot=True, fmt='.1f', cmap='Reds', ax=axes[0,0])
    axes[0,0].set_title('Missing Values by Column (%)')
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    axes[0,1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Data Types Distribution')
    
    # Memory usage by column
    memory_usage = df.memory_usage(deep=True) / 1024**2  # Convert to MB
    memory_usage = memory_usage[memory_usage > 0]  # Remove index
    axes[1,0].bar(range(len(memory_usage)), memory_usage.values)
    axes[1,0].set_title('Memory Usage by Column (MB)')
    axes[1,0].set_xlabel('Columns')
    axes[1,0].set_ylabel('Memory (MB)')
    axes[1,0].set_xticks(range(len(memory_usage)))
    axes[1,0].set_xticklabels(memory_usage.index, rotation=45)
    
    # Dataset shape info
    axes[1,1].text(0.1, 0.7, f'Rows: {df.shape[0]:,}', fontsize=14, fontweight='bold')
    axes[1,1].text(0.1, 0.5, f'Columns: {df.shape[1]}', fontsize=14, fontweight='bold')
    axes[1,1].text(0.1, 0.3, f'Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB', 
                   fontsize=14, fontweight='bold')
    axes[1,1].set_title('Dataset Statistics')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_folder / 'dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_categorical_analysis(df, output_folder):
    """Create charts for categorical variables."""
    print("Creating categorical analysis charts...")
    
    # Group analysis
    if 'group' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Categorical Variables Analysis', fontsize=16, fontweight='bold')
        
        # Top 15 groups
        group_counts = df['group'].value_counts().head(15)
        group_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Top 15 Product Groups')
        axes[0,0].set_xlabel('Group')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Group distribution pie chart (top 10)
        group_counts_top10 = df['group'].value_counts().head(10)
        axes[0,1].pie(group_counts_top10.values, labels=group_counts_top10.index, 
                     autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Top 10 Groups Distribution')
        
        # Product analysis
        if 'product' in df.columns:
            product_counts = df['product'].value_counts().head(15)
            product_counts.plot(kind='bar', ax=axes[1,0], color='lightcoral')
            axes[1,0].set_title('Top 15 Product Types')
            axes[1,0].set_xlabel('Product Type')
            axes[1,0].set_ylabel('Count')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Brand analysis (if available)
        if 'brand' in df.columns:
            brand_counts = df['brand'].dropna().value_counts().head(15)
            if len(brand_counts) > 0:
                brand_counts.plot(kind='bar', ax=axes[1,1], color='lightgreen')
                axes[1,1].set_title('Top 15 Brands')
                axes[1,1].set_xlabel('Brand')
                axes[1,1].set_ylabel('Count')
                axes[1,1].tick_params(axis='x', rotation=45)
            else:
                axes[1,1].text(0.5, 0.5, 'No brand data available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Brand Analysis')
        else:
            axes[1,1].text(0.5, 0.5, 'Brand column not available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Brand Analysis')
        
        plt.tight_layout()
        plt.savefig(output_folder / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_text_analysis(df, output_folder):
    """Create charts for text analysis."""
    print("Creating text analysis charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Text Analysis', fontsize=16, fontweight='bold')
    
    # Title length distribution
    if 'title' in df.columns:
        title_lengths = df['title'].str.len()
        axes[0,0].hist(title_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Title Length Distribution')
        axes[0,0].set_xlabel('Character Count')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(title_lengths.mean(), color='red', linestyle='--', 
                         label=f'Mean: {title_lengths.mean():.1f}')
        axes[0,0].legend()
    
    # Product name length distribution
    if 'product' in df.columns:
        product_lengths = df['product'].str.len()
        axes[0,1].hist(product_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].set_title('Product Name Length Distribution')
        axes[0,1].set_xlabel('Character Count')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(product_lengths.mean(), color='red', linestyle='--', 
                         label=f'Mean: {product_lengths.mean():.1f}')
        axes[0,1].legend()
    
    # Group name length distribution
    if 'group' in df.columns:
        group_lengths = df['group'].str.len()
        axes[1,0].hist(group_lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,0].set_title('Group Name Length Distribution')
        axes[1,0].set_xlabel('Character Count')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(group_lengths.mean(), color='red', linestyle='--', 
                         label=f'Mean: {group_lengths.mean():.1f}')
        axes[1,0].legend()
    
    # URL analysis
    if 'image_url' in df.columns:
        url_lengths = df['image_url'].str.len()
        axes[1,1].hist(url_lengths, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1,1].set_title('Image URL Length Distribution')
        axes[1,1].set_xlabel('Character Count')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(url_lengths.mean(), color='red', linestyle='--', 
                         label=f'Mean: {url_lengths.mean():.1f}')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_folder / 'text_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_entities_analysis(df, output_folder):
    """Create charts for entities analysis."""
    print("Creating entities analysis charts...")
    
    if 'entities' not in df.columns:
        print("No entities column found, skipping entities analysis...")
        return
    
    try:
        # Entity count distribution
        entity_counts = df['entities'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Entities Analysis', fontsize=16, fontweight='bold')
        
        # Entity count histogram
        axes[0,0].hist(entity_counts, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[0,0].set_title('Distribution of Entity Counts per Entry')
        axes[0,0].set_xlabel('Number of Entities')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(entity_counts.mean(), color='red', linestyle='--', 
                         label=f'Mean: {entity_counts.mean():.1f}')
        axes[0,0].legend()
        
        # Entity count box plot
        axes[0,1].boxplot(entity_counts, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0,1].set_title('Entity Count Box Plot')
        axes[0,1].set_ylabel('Number of Entities')
        
        # Most common entity types
        all_entity_names = []
        for entities in df['entities']:
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict) and 'name' in entity:
                        all_entity_names.append(entity['name'])
        
        if all_entity_names:
            entity_name_counts = Counter(all_entity_names)
            top_entities = entity_name_counts.most_common(15)
            entity_names, entity_counts_list = zip(*top_entities)
            
            axes[1,0].barh(range(len(entity_names)), entity_counts_list, color='orange')
            axes[1,0].set_yticks(range(len(entity_names)))
            axes[1,0].set_yticklabels(entity_names)
            axes[1,0].set_title('Top 15 Most Common Entity Types')
            axes[1,0].set_xlabel('Count')
        else:
            axes[1,0].text(0.5, 0.5, 'No entity data available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Entity Types Analysis')
        
        # Entity count statistics
        stats_text = f"""
        Total Entities: {entity_counts.sum():,}
        Average per Entry: {entity_counts.mean():.2f}
        Median: {entity_counts.median():.1f}
        Min: {entity_counts.min()}
        Max: {entity_counts.max()}
        Std Dev: {entity_counts.std():.2f}
        """
        axes[1,1].text(0.1, 0.5, stats_text, fontsize=12, 
                       verticalalignment='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Entity Statistics')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_folder / 'entities_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in entities analysis: {e}")

def create_correlation_analysis(df, output_folder):
    """Create correlation and relationship analysis."""
    print("Creating correlation analysis...")
    
    # Create a correlation matrix for numeric-like columns
    numeric_cols = []
    
    # Convert string lengths to numeric for correlation
    for col in df.columns:
        if col not in ['entities'] and df[col].dtype == 'object':
            try:
                lengths = df[col].str.len()
                if not lengths.isna().all():
                    numeric_cols.append(f'{col}_length')
                    df[f'{col}_length'] = lengths
            except:
                pass
    
    if len(numeric_cols) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Correlation heatmap
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0], square=True)
        axes[0].set_title('Correlation Matrix of String Lengths')
        
        # Pairwise scatter plots (first 4 columns)
        if len(numeric_cols) >= 2:
            from itertools import combinations
            pairs = list(combinations(numeric_cols[:4], 2))[:4]  # Take first 4 pairs
            
            if pairs:
                for i, (col1, col2) in enumerate(pairs):
                    if i < 4:  # Only plot first 4 pairs
                        row = i // 2
                        col = i % 2
                        if len(axes) > 1:
                            axes[1].scatter(df[col1], df[col2], alpha=0.5, s=1)
                            axes[1].set_xlabel(col1)
                            axes[1].set_ylabel(col2)
                            axes[1].set_title('String Length Relationships')
        
        plt.tight_layout()
        plt.savefig(output_folder / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_report(df, output_folder):
    """Create a summary report with key statistics."""
    print("Creating summary report...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate key statistics
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    
    # Missing values summary
    missing_summary = df.isnull().sum()
    missing_percent = (missing_summary / len(df)) * 100
    
    # Text statistics
    title_stats = ""
    if 'title' in df.columns:
        title_lengths = df['title'].str.len()
        title_stats = f"""
        Title Statistics:
        - Average length: {title_lengths.mean():.1f} characters
        - Min length: {title_lengths.min()}
        - Max length: {title_lengths.max()}
        - Unique titles: {df['title'].nunique():,}
        """
    
    # Entity statistics
    entity_stats = ""
    if 'entities' in df.columns:
        try:
            entity_counts = df['entities'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            entity_stats = f"""
        Entity Statistics:
        - Average entities per entry: {entity_counts.mean():.2f}
        - Total entities: {entity_counts.sum():,}
        - Max entities in one entry: {entity_counts.max()}
        """
        except:
            entity_stats = "Entity statistics not available"
    
    # Create summary text
    summary_text = f"""
    DATASET SUMMARY REPORT
    =====================
    
    Basic Information:
    - Total Rows: {total_rows:,}
    - Total Columns: {total_cols}
    - Memory Usage: {memory_usage:.1f} MB
    - Data Types: {df.dtypes.value_counts().to_dict()}
    
    Missing Values:
    {chr(10).join([f"- {col}: {missing_summary[col]:,} ({missing_percent[col]:.1f}%)" for col in missing_summary.index if missing_summary[col] > 0])}
    
    {title_stats}
    
    {entity_stats}
    
    Column Information:
    {chr(10).join([f"- {col}: {df[col].dtype}" for col in df.columns])}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.title('Dataset Summary Report', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_folder / 'summary_report.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to create all EDA charts."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_folder = project_root / "data"
    output_folder = project_root / "output"
    
    # Ensure output folder exists
    output_folder.mkdir(exist_ok=True)
    
    print("Creating comprehensive EDA charts...")
    print(f"Data folder: {data_folder}")
    print(f"Output folder: {output_folder}")
    
    try:
        # Load dataset
        df = load_dataset(data_folder)
        print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Create all charts
        create_basic_overview_charts(df, output_folder)
        create_categorical_analysis(df, output_folder)
        create_text_analysis(df, output_folder)
        create_entities_analysis(df, output_folder)
        create_correlation_analysis(df, output_folder)
        create_summary_report(df, output_folder)
        
        print("\n✅ All EDA charts created successfully!")
        print("📊 Charts saved to output folder:")
        print("   - dataset_overview.png")
        print("   - categorical_analysis.png")
        print("   - text_analysis.png")
        print("   - entities_analysis.png")
        print("   - correlation_analysis.png")
        print("   - summary_report.png")
        
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
