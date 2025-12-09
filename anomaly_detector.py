import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import streamlit as st

class AnomalyDetector:
    """
    Class for detecting various types of anomalies in invoice data
    """
    
    def __init__(self, rolling_months=9, sigma_threshold=3.0):
        """
        Initialize the anomaly detector
        
        Args:
            rolling_months (int): Number of months for rolling window analysis
            sigma_threshold (float): Standard deviation threshold for statistical anomalies
        """
        self.rolling_months = rolling_months
        self.sigma_threshold = sigma_threshold
        self.current_date = datetime.now()
    
    def detect_all_anomalies(self, df):
        """
        Detect all types of anomalies in the invoice data
        
        Args:
            df: Invoice dataframe
            
        Returns:
            dict: Dictionary containing indices of different anomaly types
        """
        anomalies = {
            'future_dated': self.detect_future_dated_invoices(df),
            'duplicates': self.detect_duplicate_invoices(df),
            'statistical': self.detect_statistical_anomalies(df)
        }
        
        return anomalies
    
    def detect_future_dated_invoices(self, df):
        """
        Detect invoices with future dates
        
        Args:
            df: Invoice dataframe
            
        Returns:
            list: Indices of future-dated invoices
        """
        future_mask = df['invoice_date'] > self.current_date
        future_indices = df[future_mask].index.tolist()
        
        return future_indices
    
    def detect_duplicate_invoices(self, df):
        """
        Detect potential duplicate invoices using multiple criteria
        
        Args:
            df: Invoice dataframe
            
        Returns:
            list: Indices of potential duplicate invoices
        """
        duplicate_indices = []
        
        # Method 1: Exact duplicates (invoice_number, vendor, amount)
        exact_duplicates = df.duplicated(subset=['invoice_number', 'vendor', 'amount'], keep=False)
        duplicate_indices.extend(df[exact_duplicates].index.tolist())
        
        # Method 2: Same invoice number and vendor but different amounts (suspicious)
        invoice_vendor_groups = df.groupby(['invoice_number', 'vendor'])
        for name, group in invoice_vendor_groups:
            if len(group) > 1:
                # Check if amounts are significantly different
                amounts = group['amount'].values
                if len(set(amounts)) > 1:  # Different amounts for same invoice/vendor
                    duplicate_indices.extend(group.index.tolist())
        
        # Method 3: Same vendor, amount, and date (different invoice numbers)
        vendor_amount_date_groups = df.groupby(['vendor', 'amount', df['invoice_date'].dt.date])
        for name, group in vendor_amount_date_groups:
            if len(group) > 1:
                duplicate_indices.extend(group.index.tolist())
        
        # Method 4: Very similar amounts from same vendor on same day
        df_sorted = df.sort_values(['vendor', 'invoice_date', 'amount'])
        for vendor in df['vendor'].unique():
            vendor_data = df_sorted[df_sorted['vendor'] == vendor].copy()
            
            for date in vendor_data['invoice_date'].dt.date.unique():
                date_data = vendor_data[vendor_data['invoice_date'].dt.date == date]
                
                if len(date_data) > 1:
                    amounts = date_data['amount'].values
                    
                    # Check for amounts within 1% of each other
                    for i in range(len(amounts)):
                        for j in range(i + 1, len(amounts)):
                            diff_pct = abs(amounts[i] - amounts[j]) / max(amounts[i], amounts[j])
                            if diff_pct < 0.01:  # Within 1%
                                duplicate_indices.extend(date_data.iloc[[i, j]].index.tolist())
        
        # Remove duplicates from the list itself
        duplicate_indices = list(set(duplicate_indices))
        
        return duplicate_indices
    
    def detect_statistical_anomalies(self, df):
        """
        Detect statistical anomalies using rolling window analysis and 3-sigma rule
        
        Args:
            df: Invoice dataframe
            
        Returns:
            list: Indices of statistical anomalies
        """
        anomaly_indices = []
        
        # Sort by date for rolling window analysis
        df_sorted = df.sort_values('invoice_date').copy()
        df_sorted['rolling_mean'] = np.nan
        df_sorted['rolling_std'] = np.nan
        df_sorted['z_score'] = np.nan
        
        # Calculate rolling statistics for each vendor separately
        for vendor in df['vendor'].unique():
            vendor_mask = df_sorted['vendor'] == vendor
            vendor_data = df_sorted[vendor_mask].copy()
            
            if len(vendor_data) < 5:  # Skip if not enough data
                continue
            
            # Calculate rolling statistics
            rolling_window = f"{self.rolling_months * 30}D"  # Convert months to days
            
            vendor_data['rolling_mean'] = vendor_data['amount'].rolling(
                window=rolling_window, min_periods=3
            ).mean()
            
            vendor_data['rolling_std'] = vendor_data['amount'].rolling(
                window=rolling_window, min_periods=3
            ).std()
            
            # Calculate z-scores
            valid_stats = (vendor_data['rolling_mean'].notna()) & (vendor_data['rolling_std'].notna()) & (vendor_data['rolling_std'] > 0)
            
            vendor_data.loc[valid_stats, 'z_score'] = (
                vendor_data.loc[valid_stats, 'amount'] - vendor_data.loc[valid_stats, 'rolling_mean']
            ) / vendor_data.loc[valid_stats, 'rolling_std']
            
            # Find anomalies (beyond sigma threshold)
            anomalies = vendor_data[abs(vendor_data['z_score']) > self.sigma_threshold]
            anomaly_indices.extend(anomalies.index.tolist())
            
            # Update the main dataframe
            df_sorted.loc[vendor_mask, ['rolling_mean', 'rolling_std', 'z_score']] = vendor_data[['rolling_mean', 'rolling_std', 'z_score']]
        
        # Additional method: Global statistical analysis
        global_anomalies = self._detect_global_statistical_anomalies(df)
        anomaly_indices.extend(global_anomalies)
        
        # Remove duplicates
        anomaly_indices = list(set(anomaly_indices))
        
        return anomaly_indices
    
    def _detect_global_statistical_anomalies(self, df):
        """
        Detect global statistical anomalies across all invoices
        
        Args:
            df: Invoice dataframe
            
        Returns:
            list: Indices of global statistical anomalies
        """
        anomaly_indices = []
        
        # Method 1: Tukey's method (IQR-based outliers)
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
        anomaly_indices.extend(outliers.index.tolist())
        
        # Method 2: Modified Z-score using median absolute deviation
        median = df['amount'].median()
        mad = np.median(np.abs(df['amount'] - median))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (df['amount'] - median) / mad
            outliers = df[abs(modified_z_scores) > 3.5]
            anomaly_indices.extend(outliers.index.tolist())
        
        # Method 3: Isolation Forest approach (simplified)
        # For very large amounts compared to typical invoices
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        
        # Invoices more than 5 standard deviations from mean
        extreme_outliers = df[abs(df['amount'] - mean_amount) > 5 * std_amount]
        anomaly_indices.extend(extreme_outliers.index.tolist())
        
        return list(set(anomaly_indices))
    
    def get_anomaly_summary(self, df, anomalies):
        """
        Generate summary statistics for detected anomalies
        
        Args:
            df: Invoice dataframe
            anomalies: Dictionary of anomaly indices
            
        Returns:
            dict: Summary of anomalies
        """
        summary = {
            'total_invoices': len(df),
            'anomaly_counts': {
                'future_dated': len(anomalies['future_dated']),
                'duplicates': len(anomalies['duplicates']),
                'statistical': len(anomalies['statistical'])
            },
            'anomaly_percentages': {},
            'financial_impact': {}
        }
        
        # Calculate percentages
        for anomaly_type, indices in anomalies.items():
            summary['anomaly_percentages'][anomaly_type] = (len(indices) / len(df)) * 100
        
        # Calculate financial impact
        for anomaly_type, indices in anomalies.items():
            if indices:
                anomalous_invoices = df.loc[indices]
                summary['financial_impact'][anomaly_type] = {
                    'total_amount': anomalous_invoices['amount'].sum(),
                    'average_amount': anomalous_invoices['amount'].mean(),
                    'max_amount': anomalous_invoices['amount'].max(),
                    'affected_vendors': anomalous_invoices['vendor'].nunique()
                }
            else:
                summary['financial_impact'][anomaly_type] = {
                    'total_amount': 0,
                    'average_amount': 0,
                    'max_amount': 0,
                    'affected_vendors': 0
                }
        
        return summary
    
    def explain_anomaly(self, df, invoice_index, anomaly_type):
        """
        Generate detailed explanation for a specific anomaly
        
        Args:
            df: Invoice dataframe
            invoice_index: Index of the anomalous invoice
            anomaly_type: Type of anomaly
            
        Returns:
            str: Detailed explanation of the anomaly
        """
        if invoice_index not in df.index:
            return "Invoice not found in dataset."
        
        invoice = df.loc[invoice_index]
        
        explanations = {
            'future_dated': self._explain_future_dated(invoice),
            'duplicates': self._explain_duplicate(df, invoice, invoice_index),
            'statistical': self._explain_statistical(df, invoice, invoice_index)
        }
        
        return explanations.get(anomaly_type, "Unknown anomaly type.")
    
    def _explain_future_dated(self, invoice):
        """Explain future-dated anomaly"""
        days_future = (invoice['invoice_date'] - self.current_date).days
        return f"This invoice is dated {days_future} days in the future ({invoice['invoice_date'].strftime('%Y-%m-%d')}), which is unusual and may indicate a data entry error."
    
    def _explain_duplicate(self, df, invoice, invoice_index):
        """Explain duplicate anomaly"""
        # Find similar invoices
        similar = df[
            (df['vendor'] == invoice['vendor']) & 
            (df['invoice_number'] == invoice['invoice_number']) |
            ((df['vendor'] == invoice['vendor']) & 
             (abs(df['amount'] - invoice['amount']) < 0.01) &
             (df['invoice_date'].dt.date == invoice['invoice_date'].date()))
        ]
        
        similar = similar.drop(invoice_index, errors='ignore')
        
        if len(similar) > 0:
            return f"This invoice appears to be a duplicate. Found {len(similar)} similar invoice(s) with the same vendor and/or invoice details."
        
        return "This invoice was flagged as a potential duplicate based on similarity criteria."
    
    def _explain_statistical(self, df, invoice, invoice_index):
        """Explain statistical anomaly"""
        vendor_invoices = df[df['vendor'] == invoice['vendor']]
        
        if len(vendor_invoices) > 1:
            vendor_mean = vendor_invoices['amount'].mean()
            vendor_std = vendor_invoices['amount'].std()
            
            if vendor_std > 0:
                z_score = (invoice['amount'] - vendor_mean) / vendor_std
                return f"This invoice amount (${invoice['amount']:,.2f}) is {abs(z_score):.2f} standard deviations away from the average for {invoice['vendor']} (${vendor_mean:,.2f}), making it a statistical outlier."
        
        # Global comparison
        global_mean = df['amount'].mean()
        global_std = df['amount'].std()
        
        if global_std > 0:
            global_z_score = (invoice['amount'] - global_mean) / global_std
            return f"This invoice amount (${invoice['amount']:,.2f}) is {abs(global_z_score):.2f} standard deviations away from the overall average (${global_mean:,.2f})."
        
        return "This invoice was identified as a statistical anomaly based on amount analysis."
