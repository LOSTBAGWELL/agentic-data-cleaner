"""
Agentic Data Cleaner - Core Engine
Intelligent data cleaning with human-in-the-loop reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import json


@dataclass
class CleaningStep:
    """Represents a single cleaning operation"""
    step_id: int
    action: str  # 'drop', 'impute', 'encode', 'scale', 'engineer', 'clip'
    target: str  # column name or feature
    reasoning: str
    impact: str
    confidence: float
    alternatives: List[Dict] = None
    metadata: Dict = None


class AgenticDataCleaner:
    """
    Intelligent data cleaning system that reasons with the user
    before making any changes to the data.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.analysis = {}
        self.cleaning_plan = []
        self.executed_steps = []
        self.user_confirmations = []
        self.target_column = None
        self.problem_type = None
        
    def detect_target_column(self) -> Dict:
        """
        Intelligently detect the most likely target column
        """
        candidates = []
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            unique_ratio = unique_count / total_count
            
            # Score each column
            score = 0
            reasons = []
            
            # Binary columns (likely classification targets)
            if unique_count == 2:
                score += 50
                reasons.append(f"Binary column ({self.df[col].unique()})")
            
            # Small number of categories (likely classification)
            elif 2 < unique_count <= 10 and unique_ratio < 0.1:
                score += 40
                reasons.append(f"Small categories ({unique_count} classes)")
            
            # Named like a target
            target_names = ['target', 'label', 'class', 'survived', 'outcome', 
                          'prediction', 'result', 'status', 'churn', 'fraud']
            if any(name in col.lower() for name in target_names):
                score += 30
                reasons.append("Name suggests it's a target")
            
            # Not an ID column
            if 'id' in col.lower() and unique_ratio > 0.95:
                score -= 40
                reasons.append("Likely an ID column (high uniqueness)")
            
            # Not a name column
            if 'name' in col.lower() and unique_ratio > 0.8:
                score -= 30
                reasons.append("Likely a name column (high uniqueness)")
            
            # Numeric continuous (could be regression target)
            if pd.api.types.is_numeric_dtype(self.df[col]) and unique_count > 20:
                score += 20
                reasons.append("Continuous numeric (potential regression target)")
            
            if score > 0:
                candidates.append({
                    'column': col,
                    'score': score,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'dtype': str(self.df[col].dtype),
                    'reasons': reasons
                })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'top_candidate': candidates[0] if candidates else None,
            'all_candidates': candidates[:3],  # Top 3
            'total_analyzed': len(self.df.columns)
        }
    
    def sanity_check_target(self, target_col: str) -> Dict:
        """
        Perform sanity checks on the target column
        """
        if target_col not in self.df.columns:
            return {
                'valid': False,
                'error': f"Column '{target_col}' not found in dataframe"
            }
        
        checks = []
        warnings = []
        errors = []
        
        target_data = self.df[target_col]
        
        # Check 1: Missing values
        missing_count = target_data.isna().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        
        if missing_count == 0:
            checks.append({
                'check': 'Missing Values',
                'status': 'pass',
                'message': f'No missing values (100% complete)',
                'icon': '✅'
            })
        elif missing_pct < 5:
            checks.append({
                'check': 'Missing Values',
                'status': 'warning',
                'message': f'{missing_count} missing ({missing_pct:.1f}%)',
                'icon': '⚠️'
            })
            warnings.append(f"Target has {missing_count} missing values")
        else:
            checks.append({
                'check': 'Missing Values',
                'status': 'fail',
                'message': f'{missing_count} missing ({missing_pct:.1f}%) - Too many!',
                'icon': '❌'
            })
            errors.append(f"Target has {missing_pct:.1f}% missing values")
        
        # Check 2: Data type and problem type detection
        unique_count = target_data.nunique()
        
        if unique_count == 2:
            self.problem_type = 'binary_classification'
            checks.append({
                'check': 'Problem Type',
                'status': 'pass',
                'message': f'Binary Classification (2 classes: {sorted(target_data.dropna().unique())})',
                'icon': '🎯'
            })
        elif 2 < unique_count <= 20:
            self.problem_type = 'multiclass_classification'
            checks.append({
                'check': 'Problem Type',
                'status': 'pass',
                'message': f'Multiclass Classification ({unique_count} classes)',
                'icon': '🎯'
            })
        elif pd.api.types.is_numeric_dtype(target_data) and unique_count > 20:
            self.problem_type = 'regression'
            checks.append({
                'check': 'Problem Type',
                'status': 'pass',
                'message': f'Regression (continuous values)',
                'icon': '📈'
            })
        else:
            self.problem_type = 'unknown'
            checks.append({
                'check': 'Problem Type',
                'status': 'warning',
                'message': f'Unclear ({unique_count} unique values)',
                'icon': '❓'
            })
            warnings.append("Could not clearly determine problem type")
        
        # Persist target if valid
        if len(errors) == 0:
            self.target_column = target_col
        
        # Summary stats for app metrics and insights
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'target_unique_count': unique_count,
            'target_missing_count': missing_count,
            'target_missing_pct': missing_pct
        }
        
        return {
            'valid': len(errors) == 0,
            'problem_type': self.problem_type,
            'checks': checks,
            'warnings': warnings,
            'errors': errors,
            'summary': summary
        }
    
    # analyze_background method
    def analyze_background(self) -> Dict:
        """
        Perform background analysis on the dataframe for cleaning plan
        """
        analysis = {
            'shape': self.df.shape,
            'missing_values': {},
            'unique_values': {},
            'outliers': {},
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'correlations': {}
        }
        
        # Missing values
        missing = self.df.isna().sum()
        total_rows = len(self.df)
        for col in self.df.columns:
            count = missing[col]
            if count > 0:
                pct = (count / total_rows) * 100
                severity = 'high' if pct > 50 else 'medium' if pct > 10 else 'low'
                analysis['missing_values'][col] = {'count': count, 'percentage': pct, 'severity': severity}
        
        # Unique values
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_ratio = unique_count / total_rows
            is_identifier = unique_ratio > 0.95
            is_constant = unique_count == 1
            analysis['unique_values'][col] = {
                'unique_count': unique_count,
                'unique_ratio': unique_ratio,
                'is_identifier': is_identifier,
                'is_constant': is_constant
            }
        
        # Outliers for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].nunique() > 1:  # Skip constants
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                pct = (outliers / total_rows) * 100
                if outliers > 0:
                    analysis['outliers'][col] = {
                        'count': outliers,
                        'percentage': pct,
                        'bounds': {'lower': lower, 'upper': upper}
                    }
                    
        if self.target_column and self.target_column in self.df.columns:
            # Only correlate numeric columns
            numeric_df = self.df.select_dtypes(include=[np.number])
            if self.target_column in numeric_df.columns:
                corrs = numeric_df.corr()[self.target_column].drop(self.target_column)
                analysis['correlations'] = corrs.to_dict()
        
        self.analysis = analysis
        return analysis
    
    def generate_cleaning_plan(self) -> List[CleaningStep]:
        """
        Generate a reasoned cleaning plan based on analysis
        """
        plan = []
        step_id = 1
        if not self.analysis:
            self.analysis = self.analyze_background()
        
        planned_drops = set()
        
        # Drop columns with too many missing values
        for col, info in self.analysis['missing_values'].items():
            if info['percentage'] > 80:
                plan.append(CleaningStep(
                    step_id=step_id,
                    action='drop',
                    target=col,
                    reasoning=f"Too many missing values ({info['percentage']:.1f}%) - not useful",
                    impact=f"Remove column, lose {info['count']} missing entries",
                    confidence=0.95,
                    alternatives=[{'method': 'drop_rows', 'reason': f"Lose {info['count']} rows instead"}]
                ))
                planned_drops.add(col)
                step_id += 1
        
        # Drop identifier columns
        for col, info in self.analysis['unique_values'].items():
            if info['is_identifier'] and col != self.target_column:
                plan.append(CleaningStep(
                    step_id=step_id,
                    action='drop',
                    target=col,
                    reasoning="Unique identifiers don't help in modeling",
                    impact="Remove unique ID column",
                    confidence=0.99
                ))
                planned_drops.add(col)
                step_id += 1
        
        # Drop constant columns
        for col, info in self.analysis['unique_values'].items():
            if info['is_constant'] and col != self.target_column:
                plan.append(CleaningStep(
                    step_id=step_id,
                    action='drop',
                    target=col,
                    reasoning="Constant value - no information gain",
                    impact="Remove redundant column",
                    confidence=0.99
                ))
                planned_drops.add(col)
                step_id += 1
        
        # Impute missing values (skip if planned drop)
        for col, info in self.analysis['missing_values'].items():
            if col not in planned_drops and 0 < info['percentage'] <= 80 and col != self.target_column:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = self.df[col].median()
                    mean_val = self.df[col].mean()
                    
                    alternatives = [
                        {'method': 'median', 'value': median_val, 'reason': 'Robust to outliers'},
                        {'method': 'mean', 'value': mean_val, 'reason': 'Preserves average'},
                        {'method': 'drop_rows', 'rows_lost': info['count'], 'reason': f"Lose {info['count']} rows"}
                    ]
                    
                    plan.append(CleaningStep(
                        step_id=step_id,
                        action='impute',
                        target=col,
                        reasoning=f"{info['percentage']:.1f}% missing, median={median_val:.1f} is robust",
                        impact=f"Fill {info['count']} missing values",
                        confidence=0.85,
                        alternatives=alternatives,
                        metadata={'missing_pct': info['percentage'], 'method': 'median'}
                    ))
                else:
                    mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                    
                    alternatives = [
                        {'method': 'mode', 'value': mode_val, 'reason': 'Most common value'},
                        {'method': 'new_category', 'value': 'Missing', 'reason': 'Explicit missing indicator'},
                        {'method': 'drop_rows', 'rows_lost': info['count'], 'reason': f"Lose {info['count']} rows"}
                    ]
                    
                    plan.append(CleaningStep(
                        step_id=step_id,
                        action='impute',
                        target=col,
                        reasoning=f"{info['percentage']:.1f}% missing, mode='{mode_val}' is most common",
                        impact=f"Fill {info['count']} missing values",
                        confidence=0.75,
                        alternatives=alternatives,
                        metadata={'missing_pct': info['percentage'], 'method': 'mode'}
                    ))
                
                step_id += 1
        
        # Encode categorical variables
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in planned_drops and col != self.target_column:
                unique_count = self.df[col].nunique()
                if unique_count < 2:
                    continue  # Skip: No variance, nothing to encode
                elif unique_count == 2:
                    plan.append(CleaningStep(
                        step_id=step_id,
                        action='encode',
                        target=col,
                        reasoning=f"Binary categorical - perfect for label encoding",
                        impact=f"Convert to 0/1",
                        confidence=0.99,
                        metadata={'encoding_type': 'label', 'unique_count': unique_count}
                    ))
                elif unique_count <= 10:
                    plan.append(CleaningStep(
                        step_id=step_id,
                        action='encode',
                        target=col,
                        reasoning=f"{unique_count} categories - one-hot encoding recommended",
                        impact=f"Create {unique_count - 1} binary columns",  # Adjusted for drop_first=True
                        confidence=0.90,
                        alternatives=[
                            {'method': 'onehot', 'new_cols': unique_count - 1, 'reason': 'Standard for ML'},
                            {'method': 'label', 'new_cols': 1, 'reason': 'Simpler, but assumes order'}
                        ],
                        metadata={'encoding_type': 'onehot', 'unique_count': unique_count}
                    ))
                step_id += 1
                
        # Handle outliers (if regression or high outliers)
        for col, info in self.analysis['outliers'].items():
            if col not in planned_drops and info['percentage'] > 5 and col != self.target_column:
                plan.append(CleaningStep(
                    step_id=step_id,
                    action='clip',
                    target=col,
                    reasoning=f"{info['percentage']:.1f}% outliers - clipping to IQR bounds to reduce noise",
                    impact=f"Clip {info['count']} values; preserves rows",
                    confidence=0.70 if self.problem_type == 'regression' else 0.60,
                    alternatives=[
                        {'method': 'clip', 'reason': 'Winsorize outliers'},
                        {'method': 'drop_rows', 'rows_lost': info['count'], 'reason': 'Remove outlier rows'}
                    ],
                    metadata={'method': 'clip', 'bounds': info['bounds']}
                ))
                step_id += 1
        
        self.cleaning_plan = plan
        return plan
    
    def execute_step(self, step: CleaningStep) -> Tuple[pd.DataFrame, str]:
        """
        Execute a single cleaning step
        """
        result_df = self.df.copy()
        message = ""
        
        if step.action == 'drop':
            result_df = result_df.drop(columns=[step.target])
            message = f"✅ Dropped column '{step.target}'"
        
        elif step.action == 'impute':
            if step.metadata['method'] == 'median':
                fill_value = result_df[step.target].median()
                result_df[step.target] = result_df[step.target].fillna(fill_value)  # Direct assign
                message = f"✅ Imputed '{step.target}' with median ({fill_value:.2f})"
            
            elif step.metadata['method'] == 'mean':
                fill_value = result_df[step.target].mean()
                result_df[step.target] = result_df[step.target].fillna(fill_value)
                message = f"✅ Imputed '{step.target}' with mean ({fill_value:.2f})"
            
            elif step.metadata['method'] == 'mode':
                fill_value = result_df[step.target].mode()[0]
                result_df[step.target] = result_df[step.target].fillna(fill_value)
                message = f"✅ Imputed '{step.target}' with mode ('{fill_value}')"
        
        elif step.action == 'encode':
            if step.metadata['encoding_type'] == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                result_df[step.target] = le.fit_transform(result_df[step.target].astype(str))
                message = f"✅ Label encoded '{step.target}' ({le.classes_})"
            
            elif step.metadata['encoding_type'] == 'onehot':
                dummies = pd.get_dummies(result_df[step.target], prefix=step.target, drop_first=True)
                result_df = pd.concat([result_df.drop(columns=[step.target]), dummies], axis=1)
                message = f"✅ One-hot encoded '{step.target}' (created {len(dummies.columns)} columns)"
                
        elif step.action == 'clip':
            lower, upper = step.metadata['bounds']['lower'], step.metadata['bounds']['upper']
            result_df[step.target] = result_df[step.target].clip(lower, upper)
            message = f"✅ Clipped '{step.target}' outliers to [{lower:.2f}, {upper:.2f}]"
        
        self.df = result_df
        self.executed_steps.append(step)
        
        return result_df, message
    
    # Preview method for interactive "what if" in app
    def preview_step(self, step: CleaningStep, sample_size: int = 5) -> pd.DataFrame:
        """
        Simulate a step on a small sample without modifying main df
        """
        preview_df = self.df.head(sample_size).copy()
        # Reuse execute logic but on preview_df (no message needed)
        if step.action == 'drop':
            preview_df = preview_df.drop(columns=[step.target])
        
        elif step.action == 'impute':
            if step.metadata['method'] == 'median':
                fill_value = preview_df[step.target].median()
                preview_df[step.target] = preview_df[step.target].fillna(fill_value)
            # Similarly for mean/mode
        
        elif step.action == 'encode':
            if step.metadata['encoding_type'] == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                preview_df[step.target] = le.fit_transform(preview_df[step.target].astype(str))
            elif step.metadata['encoding_type'] == 'onehot':
                dummies = pd.get_dummies(preview_df[step.target], prefix=step.target, drop_first=True)
                preview_df = pd.concat([preview_df.drop(columns=[step.target]), dummies], axis=1)
        
        elif step.action == 'clip':
            lower, upper = step.metadata['bounds']['lower'], step.metadata['bounds']['upper']
            preview_df[step.target] = preview_df[step.target].clip(lower, upper)
        
        return preview_df
    
    def get_cleaning_summary(self) -> Dict:
        """
        Get a summary of all cleaning operations
        """
        summary = {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'steps_executed': len(self.executed_steps),
            'rows_changed': len(self.original_df) - len(self.df),
            'columns_changed': len(self.original_df.columns) - len(self.df.columns),
            'executed_steps': [
                {
                    'step_id': step.step_id,
                    'action': step.action,
                    'target': step.target,
                    'reasoning': step.reasoning
                }
                for step in self.executed_steps
            ]
        }
        # Add beginner-friendly insights
        summary['insights'] = []
        if summary['rows_changed'] > 0:
            summary['insights'].append(f"You removed {summary['rows_changed']} rows— this helps make your data more reliable by eliminating incomplete entries.")
        if summary['columns_changed'] > 0:
            summary['insights'].append(f"You simplified your dataset by dropping {summary['columns_changed']} columns, focusing on what's important for analysis.")
        return summary


# Helper functions
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


if __name__ == "__main__":
    # Quick test
    print("✅ Aka Cleaner Engine Ready!")
    print("\nUsage:")
    print("  from agentic_data_cleaner import AgenticDataCleaner")
    print("  cleaner = AgenticDataCleaner(df)")
    print("  target_detection = cleaner.detect_target_column()")
    print("  sanity_check = cleaner.sanity_check_target('Survived')")
    print("  plan = cleaner.generate_cleaning_plan()")
