#!/usr/bin/env python3
"""
Comprehensive EDA Script
Performs complete Exploratory Data Analysis with both statistical analysis and visualizations.
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

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Set style
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
            if i % 50 == 0:
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

def perform_statistical_analysis(df, output_file):
    """Perform comprehensive statistical analysis."""
    def write_and_print(text):
        print(text)
        output_file.write(text + "\n")
    
    write_and_print("="*80)
    write_and_print("COMPREHENSIVE STATISTICAL ANALYSIS")
    write_and_print("="*80)
    
    # Basic dataset info
    write_and_print(f"\n📊 DATASET OVERVIEW")
    write_and_print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    write_and_print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    write_and_print(f"Data Types: {dict(df.dtypes.value_counts())}")
    
    # Missing values analysis
    write_and_print(f"\n🔍 MISSING VALUES ANALYSIS")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)
    
    if len(missing_df) > 0:
        write_and_print(missing_df.to_string())
    else:
        write_and_print("✅ No missing values found!")
    
    # Column analysis
    write_and_print(f"\n📋 COLUMN ANALYSIS")
    for col in df.columns:
        write_and_print(f"\n{col}:")
        write_and_print(f"  Type: {df[col].dtype}")
        
        # Safe unique count calculation
        try:
            unique_count = df[col].nunique()
            write_and_print(f"  Unique values: {unique_count:,}")
        except:
            write_and_print(f"  Unique values: Cannot calculate (complex data type)")
        
        write_and_print(f"  Missing: {df[col].isnull().sum():,} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        
        if df[col].dtype == 'object':
            # String analysis
            try:
                lengths = df[col].str.len()
                write_and_print(f"  String length - Min: {lengths.min()}, Max: {lengths.max()}, Avg: {lengths.mean():.1f}")
            except:
                write_and_print(f"  String length analysis not available (complex data type)")
            
            # Most frequent values - skip for entities column
            if col != 'entities':
                try:
                    most_frequent = df[col].value_counts().head(3)
                    write_and_print(f"  Most frequent values:")
                    for val, count in most_frequent.items():
                        write_and_print(f"    '{val}': {count:,} ({count/len(df)*100:.1f}%)")
                except:
                    write_and_print(f"  Most frequent values analysis not available")
            else:
                write_and_print(f"  Most frequent values: Complex data structure (list of dicts)")
    
    # Entities analysis
    if 'entities' in df.columns:
        write_and_print(f"\n🏷️ ENTITIES ANALYSIS")
        try:
            # Safe entity count calculation
            entity_counts = []
            for entities in df['entities']:
                if isinstance(entities, list):
                    entity_counts.append(len(entities))
                else:
                    entity_counts.append(0)
            
            entity_counts = pd.Series(entity_counts)
            write_and_print(f"  Average entities per entry: {entity_counts.mean():.2f}")
            write_and_print(f"  Min entities: {entity_counts.min()}")
            write_and_print(f"  Max entities: {entity_counts.max()}")
            write_and_print(f"  Total entities: {entity_counts.sum():,}")
            
            # Most common entity types
            all_entity_names = []
            for entities in df['entities']:
                if isinstance(entities, list):
                    for entity in entities:
                        if isinstance(entity, dict) and 'name' in entity:
                            all_entity_names.append(entity['name'])
            
            if all_entity_names:
                entity_name_counts = Counter(all_entity_names)
                write_and_print(f"  Most common entity types:")
                for name, count in entity_name_counts.most_common(10):
                    write_and_print(f"    {name}: {count:,}")
        except Exception as e:
            write_and_print(f"  Error analyzing entities: {e}")

def create_comprehensive_charts(df, output_folder):
    """Create comprehensive visualization suite."""
    print("\n📊 Creating comprehensive visualizations...")
    
    # 1. Dataset Overview Dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Dataset Overview Dashboard', fontsize=20, fontweight='bold')
    
    # Missing values heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing %': missing_percent})
    sns.heatmap(missing_df.T, annot=True, fmt='.1f', cmap='Reds', ax=ax1)
    ax1.set_title('Missing Values by Column (%)')
    
    # Data types pie chart
    ax2 = fig.add_subplot(gs[0, 2:])
    dtype_counts = df.dtypes.value_counts()
    ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Data Types Distribution')
    
    # Memory usage
    ax3 = fig.add_subplot(gs[1, :2])
    memory_usage = df.memory_usage(deep=True) / 1024**2
    memory_usage = memory_usage[memory_usage > 0]
    ax3.bar(range(len(memory_usage)), memory_usage.values, color='skyblue')
    ax3.set_title('Memory Usage by Column (MB)')
    ax3.set_xlabel('Columns')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_xticks(range(len(memory_usage)))
    ax3.set_xticklabels(memory_usage.index, rotation=45)
    
    # Dataset statistics
    ax4 = fig.add_subplot(gs[1, 2:])
    stats_text = f"""
    Dataset Statistics:
    
    Rows: {df.shape[0]:,}
    Columns: {df.shape[1]}
    Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
    
    Missing Values:
    {chr(10).join([f"- {col}: {df[col].isnull().sum():,} ({df[col].isnull().sum()/len(df)*100:.1f}%)" for col in df.columns if df[col].isnull().sum() > 0])}
    """
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Dataset Summary')
    ax4.axis('off')
    
    # Top groups
    ax5 = fig.add_subplot(gs[2, :2])
    if 'group' in df.columns:
        group_counts = df['group'].value_counts().head(10)
        group_counts.plot(kind='bar', ax=ax5, color='lightcoral')
        ax5.set_title('Top 10 Product Groups')
        ax5.set_xlabel('Group')
        ax5.set_ylabel('Count')
        ax5.tick_params(axis='x', rotation=45)
    
    # Top products
    ax6 = fig.add_subplot(gs[2, 2:])
    if 'product' in df.columns:
        product_counts = df['product'].value_counts().head(10)
        product_counts.plot(kind='bar', ax=ax6, color='lightgreen')
        ax6.set_title('Top 10 Product Types')
        ax6.set_xlabel('Product Type')
        ax6.set_ylabel('Count')
        ax6.tick_params(axis='x', rotation=45)
    
    # Text length distributions
    ax7 = fig.add_subplot(gs[3, :2])
    if 'title' in df.columns:
        title_lengths = df['title'].str.len()
        ax7.hist(title_lengths, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax7.set_title('Title Length Distribution')
        ax7.set_xlabel('Character Count')
        ax7.set_ylabel('Frequency')
        ax7.axvline(title_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {title_lengths.mean():.1f}')
        ax7.legend()
    
    # Entity count distribution
    ax8 = fig.add_subplot(gs[3, 2:])
    if 'entities' in df.columns:
        try:
            # Safe entity count calculation
            entity_counts = []
            for entities in df['entities']:
                if isinstance(entities, list):
                    entity_counts.append(len(entities))
                else:
                    entity_counts.append(0)
            
            entity_counts = pd.Series(entity_counts)
            ax8.hist(entity_counts, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax8.set_title('Entity Count Distribution')
            ax8.set_xlabel('Number of Entities')
            ax8.set_ylabel('Frequency')
            ax8.axvline(entity_counts.mean(), color='red', linestyle='--', 
                       label=f'Mean: {entity_counts.mean():.1f}')
            ax8.legend()
        except Exception as e:
            ax8.text(0.5, 0.5, f'Entity analysis error: {str(e)[:50]}...', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Entity Count Distribution')
    
    plt.savefig(output_folder / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Categorical Analysis
    if 'group' in df.columns or 'product' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Detailed Categorical Analysis', fontsize=16, fontweight='bold')
        
        # Group analysis
        if 'group' in df.columns:
            group_counts = df['group'].value_counts().head(15)
            group_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
            axes[0,0].set_title('Top 15 Product Groups')
            axes[0,0].set_xlabel('Group')
            axes[0,0].set_ylabel('Count')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Group pie chart
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
        
        # Brand analysis
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
        
        plt.tight_layout()
        plt.savefig(output_folder / 'categorical_analysis_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Text Analysis Suite
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Text Analysis Suite', fontsize=16, fontweight='bold')
    
    # Title analysis
    if 'title' in df.columns:
        title_lengths = df['title'].str.len()
        axes[0,0].hist(title_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Title Length Distribution')
        axes[0,0].set_xlabel('Character Count')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(title_lengths.mean(), color='red', linestyle='--', 
                         label=f'Mean: {title_lengths.mean():.1f}')
        axes[0,0].legend()
    
    # Product name analysis
    if 'product' in df.columns:
        product_lengths = df['product'].str.len()
        axes[0,1].hist(product_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].set_title('Product Name Length Distribution')
        axes[0,1].set_xlabel('Character Count')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(product_lengths.mean(), color='red', linestyle='--', 
                         label=f'Mean: {product_lengths.mean():.1f}')
        axes[0,1].legend()
    
    # Group name analysis
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
    plt.savefig(output_folder / 'text_analysis_suite.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ All visualizations created successfully!")

def main():
    """Main function for comprehensive EDA."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_folder = project_root / "data"
    output_folder = project_root / "output"
    
    # Ensure output folder exists
    output_folder.mkdir(exist_ok=True)
    
    print("🚀 Starting Comprehensive EDA Analysis...")
    print(f"Data folder: {data_folder}")
    print(f"Output folder: {output_folder}")
    
    try:
        # Load dataset
        df = load_dataset(data_folder)
        print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Perform statistical analysis
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analysis_file = output_folder / f"comprehensive_eda_analysis_{timestamp.replace(':', '-')}.txt"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"Comprehensive EDA Analysis Report\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write(f"Dataset: entities_dataset_v2\n")
            f.write("="*80 + "\n\n")
            
            perform_statistical_analysis(df, f)
        
        # Create visualizations
        create_comprehensive_charts(df, output_folder)
        
        print(f"\n🎉 Comprehensive EDA completed successfully!")
        print(f"📊 Analysis report saved to: {analysis_file}")
        print(f"📈 Visualizations saved to: {output_folder}")
        print(f"\nGenerated files:")
        print(f"  - comprehensive_dashboard.png")
        print(f"  - categorical_analysis_detailed.png") 
        print(f"  - text_analysis_suite.png")
        print(f"  - comprehensive_eda_analysis_{timestamp.replace(':', '-')}.txt")
        
    except Exception as e:
        print(f"❌ Error in comprehensive EDA: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
