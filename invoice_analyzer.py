import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from pdf_processor_old import PDFInvoiceProcessor

class InvoiceAnalyzer:
    """
    Main class for loading and preprocessing invoice data from CSV, Excel, or PDF files
    """
    
    def __init__(self):
        self.required_columns = ['invoice_number', 'amount', 'vendor', 'invoice_date']
        self.pdf_processor = PDFInvoiceProcessor()
    
    def load_data(self, uploaded_files):
        """
        Load invoice data from uploaded files (CSV, Excel, or PDF)
        
        Args:
            uploaded_files: Single file or list of Streamlit uploaded file objects
            
        Returns:
            pandas.DataFrame: Cleaned and processed invoice data
        """
        try:
            # Handle single file or multiple files
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]
            
            # Separate files by type
            pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
            csv_excel_files = [f for f in uploaded_files if f.name.endswith(('.csv', '.xlsx', '.xls'))]
            
            dataframes = []
            
            # Process PDF files
            if pdf_files:
                st.info(f"Processing {len(pdf_files)} PDF file(s)...")
                pdf_df = self.pdf_processor.process_multiple_pdfs(pdf_files)
                dataframes.append(pdf_df)
            
            # Process CSV/Excel files
            for uploaded_file in csv_excel_files:
                df = None
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                
                if df is not None:
                    # Clean and validate data
                    df = self._clean_data(df)
                    dataframes.append(df)
            
            if not dataframes:
                raise ValueError("No supported files found. Please upload PDF, CSV, or Excel files.")
            
            # Combine all dataframes
            if len(dataframes) == 1:
                final_df = dataframes[0]
            else:
                final_df = pd.concat(dataframes, ignore_index=True)
            
            return final_df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _clean_data(self, df):
        """
        Clean and preprocess the invoice data
        
        Args:
            df: Raw dataframe
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        # Normalize column names (handle different naming conventions)
        df.columns = df.columns.str.lower().str.strip()
        
        # Map common column variations
        column_mapping = {
            'invoice_no': 'invoice_number',
            'invoice_num': 'invoice_number',
            'inv_number': 'invoice_number',
            'inv_no': 'invoice_number',
            'supplier': 'vendor',
            'vendor_name': 'vendor',
            'supplier_name': 'vendor',
            'invoice_amount': 'amount',
            'total_amount': 'amount',
            'amount_due': 'amount',
            'date': 'invoice_date',
            'inv_date': 'invoice_date',
            'bill_date': 'invoice_date'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data types
        df = self._clean_data_types(df)
        
        # Remove duplicates based on all columns
        initial_count = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        if len(df) < initial_count:
            st.info(f"Removed {initial_count - len(df)} exact duplicate rows")
        
        # Filter out rows with missing critical data
        df = df.dropna(subset=self.required_columns)
        
        return df
    
    def _clean_data_types(self, df):
        """
        Clean and convert data types
        
        Args:
            df: Dataframe with raw data types
            
        Returns:
            pandas.DataFrame: Dataframe with cleaned data types
        """
        # Clean invoice numbers
        df['invoice_number'] = df['invoice_number'].astype(str).str.strip()
        
        # Clean vendor names
        df['vendor'] = df['vendor'].astype(str).str.strip().str.title()
        
        # Clean amounts
        if df['amount'].dtype == 'object':
            # Remove currency symbols and commas
            df['amount'] = df['amount'].astype(str).str.replace(r'[$,]', '', regex=True)
            df['amount'] = df['amount'].str.replace(r'[()]', '', regex=True)  # Remove parentheses
        
        # Convert to numeric, handling errors
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove negative amounts and zero amounts
        df = df[df['amount'] > 0]
        
        # Clean dates
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce', dayfirst=True)
        
        # Remove rows with invalid dates or amounts
        df = df.dropna(subset=['amount', 'invoice_date'])
        
        return df
    
    def get_data_summary(self, df):
        """
        Generate summary statistics for the invoice data
        
        Args:
            df: Invoice dataframe
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_invoices': len(df),
            'unique_vendors': df['vendor'].nunique(),
            'date_range': {
                'start': df['invoice_date'].min(),
                'end': df['invoice_date'].max()
            },
            'amount_stats': {
                'total': df['amount'].sum(),
                'mean': df['amount'].mean(),
                'median': df['amount'].median(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max()
            },
            'vendors': df['vendor'].value_counts().head(10).to_dict()
        }
        
        return summary
    
    def validate_data_quality(self, df):
        """
        Validate data quality and return quality metrics
        
        Args:
            df: Invoice dataframe
            
        Returns:
            dict: Data quality metrics
        """
        quality_metrics = {
            'completeness': {
                'invoice_number': (df['invoice_number'].notna().sum() / len(df)) * 100,
                'amount': (df['amount'].notna().sum() / len(df)) * 100,
                'vendor': (df['vendor'].notna().sum() / len(df)) * 100,
                'invoice_date': (df['invoice_date'].notna().sum() / len(df)) * 100
            },
            'uniqueness': {
                'invoice_numbers': (df['invoice_number'].nunique() / len(df)) * 100
            },
            'validity': {
                'positive_amounts': (df['amount'] > 0).sum() / len(df) * 100,
                'valid_dates': df['invoice_date'].notna().sum() / len(df) * 100
            },
            'consistency': {
                'vendor_name_consistency': len(df['vendor'].unique()) / len(df) * 100
            }
        }
        
        return quality_metrics
