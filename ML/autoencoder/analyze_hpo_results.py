"""
Analyze and visualize hyperparameter search results.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_study_summary(summary_path):
    """Load study summary JSON file."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def create_results_dataframe(summary):
    """Convert study summary to pandas DataFrame."""
    trials = summary['trials']
    df = pd.DataFrame(trials)
    
    # Expand params dict into separate columns
    params_df = pd.json_normalize(df['params'])
    df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
    
    return df


def plot_optimization_history(df, output_path=None):
    """Plot validation loss over trials."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['number'], df['value'], 'o-', alpha=0.7)
    
    # Highlight best trial
    best_idx = df['value'].idxmin()
    plt.plot(df.loc[best_idx, 'number'], df.loc[best_idx, 'value'], 
             'r*', markersize=20, label=f'Best (Trial {df.loc[best_idx, "number"]})')
    
    # Running minimum
    running_min = df['value'].cummin()
    plt.plot(df['number'], running_min, 'r--', alpha=0.5, label='Running Best')
    
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Loss')
    plt.title('Optimization History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved optimization history to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_parameter_importance(df, output_path=None):
    """Plot correlation of each parameter with validation loss."""
    param_cols = [col for col in df.columns if col not in ['number', 'value', 'duration']]
    
    n_params = len(param_cols)
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, param in zip(axes, param_cols):
        # Handle categorical parameters
        if df[param].dtype == 'object' or df[param].nunique() < 10:
            df_plot = df.groupby(param)['value'].mean().sort_values()
            ax.bar(range(len(df_plot)), df_plot.values)
            ax.set_xticks(range(len(df_plot)))
            ax.set_xticklabels(df_plot.index, rotation=45)
        else:
            ax.scatter(df[param], df['value'], alpha=0.6)
            
            # Fit trend line
            z = pd.Series.corr(df[param], df['value'])
            ax.set_title(f'{param}\n(corr: {z:.3f})')
        
        ax.set_xlabel(param)
        ax.set_ylabel('Validation Loss')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter importance to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_parameter_distributions(df, output_path=None):
    """Plot distributions of hyperparameters for best vs worst trials."""
    # Get top 20% and bottom 20% trials
    n_top = max(1, len(df) // 5)
    df_sorted = df.sort_values('value')
    df_best = df_sorted.head(n_top)
    df_worst = df_sorted.tail(n_top)
    
    param_cols = [col for col in df.columns if col not in ['number', 'value', 'duration']]
    
    n_params = len(param_cols)
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, param in zip(axes, param_cols):
        if df[param].dtype == 'object' or df[param].nunique() < 10:
            # Categorical: use count plot
            best_counts = df_best[param].value_counts()
            worst_counts = df_worst[param].value_counts()
            
            x = sorted(df[param].unique())
            best_vals = [best_counts.get(val, 0) for val in x]
            worst_vals = [worst_counts.get(val, 0) for val in x]
            
            x_pos = range(len(x))
            width = 0.35
            ax.bar([p - width/2 for p in x_pos], best_vals, width, label='Best 20%', alpha=0.7)
            ax.bar([p + width/2 for p in x_pos], worst_vals, width, label='Worst 20%', alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x, rotation=45)
        else:
            # Continuous: use histogram
            ax.hist([df_best[param], df_worst[param]], bins=10, 
                   label=['Best 20%', 'Worst 20%'], alpha=0.7)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Parameter Distributions: Best vs Worst Trials', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter distributions to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_parallel_coordinate(df, output_path=None):
    """Create parallel coordinate plot."""
    # Normalize parameters to [0, 1] for visualization
    param_cols = [col for col in df.columns if col not in ['number', 'value', 'duration']]
    df_plot = df[param_cols + ['value']].copy()
    
    # Handle categorical variables
    for col in param_cols:
        if df_plot[col].dtype == 'object':
            df_plot[col] = pd.Categorical(df_plot[col]).codes
    
    # Normalize
    for col in param_cols:
        min_val = df_plot[col].min()
        max_val = df_plot[col].max()
        if max_val > min_val:
            df_plot[col] = (df_plot[col] - min_val) / (max_val - min_val)
    
    # Color by validation loss (lower is better)
    plt.figure(figsize=(12, 6))
    
    # Plot top 10 trials
    df_top = df_plot.nsmallest(10, 'value')
    
    for idx, row in df_top.iterrows():
        values = row[param_cols].values
        plt.plot(param_cols, values, 'o-', alpha=0.6, linewidth=2)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Normalized Value')
    plt.title('Parallel Coordinate Plot (Top 10 Trials)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved parallel coordinate plot to {output_path}")
    else:
        plt.show()
    plt.close()


def print_summary(summary, df):
    """Print summary statistics."""
    print("="*80)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*80)
    print(f"\nStudy Name: {summary['study_name']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Total Trials: {summary['n_trials']}")
    print(f"Complete Trials: {summary['n_complete']}")
    print(f"Pruned Trials: {summary['n_pruned']}")
    
    print(f"\n{'='*80}")
    print("BEST TRIAL")
    print("="*80)
    print(f"Validation Loss: {summary['best_value']:.6f}")
    print(f"\nBest Hyperparameters:")
    for key, value in summary['best_params'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n{'='*80}")
    print("TOP 5 TRIALS")
    print("="*80)
    df_top5 = df.nsmallest(5, 'value')
    param_cols = [col for col in df.columns if col not in ['number', 'value', 'duration']]
    
    for i, (idx, row) in enumerate(df_top5.iterrows(), 1):
        print(f"\n{i}. Trial {int(row['number'])}")
        print(f"   Val Loss: {row['value']:.6f}")
        if row['duration'] is not None:
            print(f"   Duration: {row['duration']:.1f}s")
        for col in param_cols:
            val = row[col]
            if isinstance(val, float):
                print(f"   {col}: {val:.6f}")
            else:
                print(f"   {col}: {val}")
    
    print(f"\n{'='*80}")
    print("PARAMETER STATISTICS")
    print("="*80)
    
    for col in param_cols:
        if df[col].dtype != 'object' and df[col].nunique() > 5:
            # Continuous parameter
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.6f}")
            print(f"  Std: {df[col].std():.6f}")
            print(f"  Min: {df[col].min():.6f}")
            print(f"  Max: {df[col].max():.6f}")
            
            # Best trials average
            best_n = max(1, len(df) // 10)
            best_avg = df.nsmallest(best_n, 'value')[col].mean()
            print(f"  Best 10% avg: {best_avg:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument('summary_file', type=str,
                        help='Path to study_summary_*.json file')
    parser.add_argument('--output_dir', type=str, default='hpo_plots',
                        help='Directory to save plots')
    parser.add_argument('--no_plots', action='store_true',
                        help='Only print summary, do not create plots')
    
    args = parser.parse_args()
    
    # Load summary
    print(f"Loading results from {args.summary_file}...")
    summary = load_study_summary(args.summary_file)
    df = create_results_dataframe(summary)
    
    # Print summary
    print_summary(summary, df)
    
    if not args.no_plots:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating plots in {output_dir}...")
        
        # Generate plots
        plot_optimization_history(df, output_dir / 'optimization_history.png')
        plot_parameter_importance(df, output_dir / 'parameter_importance.png')
        plot_parameter_distributions(df, output_dir / 'parameter_distributions.png')
        plot_parallel_coordinate(df, output_dir / 'parallel_coordinate.png')
        
        print(f"\nAll plots saved to {output_dir}/")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
