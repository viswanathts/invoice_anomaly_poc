import pdfplumber
import PyPDF2
import pandas as pd
import re
from datetime import datetime
import streamlit as st
from typing import Dict, List, Optional
import io
from pathlib import Path
from typing import Optional, List, Dict, Any

class PDFInvoiceProcessor:
    """
    Processes PDF invoices to extract structured data
    """
    
    def __init__(self):
        self.amount_patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $1,234.56
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*\$',  # 1,234.56$
            r'TOTAL[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # TOTAL: $1,234.56
            r'AMOUNT[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # AMOUNT: $1,234.56
            r'DUE[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # DUE: $1,234.56
        ]
        
        self.date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # MM/DD/YYYY or MM-DD-YYYY
            r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
            r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})',  # January 15, 2024
            r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})',  # 15 January 2024
        ]
        
        self.invoice_number_patterns = [
            r'INVOICE\s*#?[:\s]*([A-Z0-9\-/]+)',
            r'INV\s*#?[:\s]*([A-Z0-9\-/]+)',
            r'BILL\s*#?[:\s]*([A-Z0-9\-/]+)',
            r'REF\s*#?[:\s]*([A-Z0-9\-/]+)',
            r'NO\.\s*([A-Z0-9\-/]+)',
        ]
        
        self.vendor_patterns = [
            r'FROM[:\s]*(.+?)(?:\n|$)',
            r'BILL\s+FROM[:\s]*(.+?)(?:\n|$)',
            r'VENDOR[:\s]*(.+?)(?:\n|$)',
            r'COMPANY[:\s]*(.+?)(?:\n|$)',
        ]

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        try:
            # Method 1: pdfplumber (better for complex layouts)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                return text
                
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {str(e)}")
        
        try:
            # Method 2: PyPDF2 (fallback)
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        return text

    def extract_invoice_data(self, pdf_file, filename: str) -> Dict:
        """Extract structured data from PDF invoice"""
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            raise ValueError("Unable to extract text from PDF")
        
        # Extract individual fields
        invoice_data = {
            'filename': filename,
            'invoice_number': self._extract_invoice_number(text),
            'vendor': self._extract_vendor(text, filename),
            'amount': self._extract_amount(text),
            'invoice_date': self._extract_date(text),
            'raw_text': text[:500]  # Store first 500 chars for debugging
        }
        
        return invoice_data

    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract invoice amount with improved patterns and validation"""
        
        if not text:
            return None
        
        # Handle multi-line amount splitting (generic approach)
        lines = text.split('\n')
        reconstructed_amount = self._handle_multiline_amounts(lines)
        if reconstructed_amount:
            return reconstructed_amount
        
        # ADD DEBUG HERE - after the multiline check
        print(f"DEBUG: Processing file with text sample: {text[:300]}")
        print(f"DEBUG: Multiline handler returned: {reconstructed_amount}")

        # Clean and prepare text
        text_upper = text.upper()
        text_clean = ' '.join(text.split())  # Normalize whitespace
        
        # Primary patterns - ordered by priority and specificity
        primary_patterns = [
            # FCVB specific patterns (highest priority)
            r'TOTAL\s+AMOUNT\s+(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'NON-TAXABLE\s+(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'AMOUNT\s+CHARGED.*?ACCOUNT.*?(\d{1,3}(?:,\d{3})*\.\d{2})',
            
            # Most specific patterns
            r'(?:AMOUNT\s+DUE|TOTAL\s+DUE|BALANCE\s+DUE|AMOUNT\s+OWING)\s*[:\-]?\s*\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'(?:TOTAL\s+AMOUNT|INVOICE\s+TOTAL|GRAND\s+TOTAL)\s*[:\-]?\s*\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'(?:NET\s+TOTAL|SUBTOTAL)\s*[:\-]?\s*\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
            
            # Pattern specifically for FCVB-style invoices
            r'(\d{1,3}(?:,\d{3})*\.\d{2})\s+(?:TAXABLE|NON-TAXABLE)',
            
            # Dollar sign patterns
            r'\$\s*(\d{1,3}(?:,\d{3})*\.\d{2})(?=\s|$)',
            
            # Generic TOTAL patterns
            r'TOTAL\s*[:\-]?\s*\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
        ]
        
        # Secondary patterns for fallback
        secondary_patterns = [
            # Look for amounts near invoice keywords (wider context)
            r'(?:INVOICE|BILL|CHARGE|DUE|PAYMENT|AMOUNT)[\s\S]{0,100}?(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'(\d{1,3}(?:,\d{3})*\.\d{2})[\s\S]{0,100}?(?:INVOICE|BILL|CHARGE|DUE|PAYMENT)',
            
            # Standalone monetary amounts with proper formatting
            r'\b(\d{1,3}(?:,\d{3})*\.\d{2})\b',
        ]
        
        def extract_and_validate_amounts(patterns, text_to_search, min_amount=10, max_amount=10000000):
            """Extract amounts using patterns and validate them"""
            found_amounts = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_to_search, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                
                for match in matches:
                    try:
                        # Handle tuple matches (from groups)
                        if isinstance(match, tuple):
                            amount_str = ''.join(str(m) for m in match if m)
                        else:
                            amount_str = str(match)
                        
                        # Clean and convert - CRITICAL: properly handle commas
                        amount_str = amount_str.replace(',', '').strip()
                        amount = float(amount_str)
                        
                        # Validate amount range
                        if min_amount <= amount <= max_amount:
                            # Additional validation to avoid common false positives
                            if not self._is_likely_false_positive(amount_str, text):
                                found_amounts.append(amount)
                                
                    except (ValueError, TypeError):
                        continue
            
            return found_amounts
        
        # Try primary patterns first
        amounts = extract_and_validate_amounts(primary_patterns, text_upper)
        # ADD DEBUG HERE - after primary patterns
        print(f"DEBUG: Found amounts from primary patterns: {amounts}")

        if amounts:
            return max(amounts)  # Return the largest amount found
        
        # Try secondary patterns with original text
        amounts = extract_and_validate_amounts(secondary_patterns, text)
        if amounts:
            return max(amounts)
        
        # Last resort: look for any properly formatted currency amount
        final_pattern = r'\b(\d{1,3}(?:,\d{3})*\.\d{2})\b'
        amounts = extract_and_validate_amounts([final_pattern], text, min_amount=100, max_amount=10000000)
        
        if amounts:
            # For final fallback, prefer amounts that appear in invoice-related context
            contextual_amounts = []
            for amount in amounts:
                # Format amount string for searching (handle commas properly)
                amount_str = f"{amount:,.2f}"
                amount_str_no_comma = f"{amount:.2f}"
                
                # Check if amount appears near invoice keywords
                context_pattern = f'(?:INVOICE|BILL|TOTAL|DUE|AMOUNT|PAYMENT|CHARGE)[\s\S]{{0,100}}?(?:{re.escape(amount_str)}|{re.escape(amount_str_no_comma)})'
                if re.search(context_pattern, text, re.IGNORECASE):
                    contextual_amounts.append(amount)
            
            if contextual_amounts:
                return max(contextual_amounts)
            else:
                return max(amounts)
        
        return None

    def _handle_multiline_amounts(self, lines: List[str]) -> Optional[float]:
        """
        Generic handler for multi-line amount splitting in invoices.
        Detects when amounts are split across lines and reconstructs them.
        """
    
        # Look for patterns where amounts might be split
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # Pattern 1: Look for lines with partial amounts followed by TAXABLE/NON-TAXABLE
            if re.search(r'\d+\.\d{2}\s+(?:TAXABLE|NON-TAXABLE)', line_clean, re.IGNORECASE):
                # Extract the amount from this line
                amount_match = re.search(r'(\d+\.\d{2})\s+(?:TAXABLE|NON-TAXABLE)', line_clean, re.IGNORECASE)
                if amount_match:
                    partial_amount = amount_match.group(1)
                    
                    # Look for potential prefix digits in nearby lines
                    reconstructed = self._find_amount_prefix(lines, i, partial_amount)
                    if reconstructed:
                        return reconstructed
            
            # Pattern 2: Look for lines that end with just digits (potential amount prefix)
            if re.match(r'^\s*\d+\s*$', line_clean):
                prefix_digit = line_clean.strip()
                
                # Look in next few lines for amount with decimal
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_line = lines[j].strip()
                    
                    # Check if next line has amount with TAXABLE/NON-TAXABLE
                    amount_match = re.search(r'(\d+\.\d{2})\s+(?:TAXABLE|NON-TAXABLE)', next_line, re.IGNORECASE)
                    if amount_match:
                        partial_amount = amount_match.group(1)
                        try:
                            reconstructed_amount = float(prefix_digit + partial_amount)
                            if 100 <= reconstructed_amount <= 10000000:  # Reasonable invoice range
                                return reconstructed_amount
                        except ValueError:
                            continue
            
            # Pattern 3: Look for "NON-TAXABLE" followed by amount on next line (FCVB specific)
            if 'NON-TAXABLE' in line_clean.upper():
                # Check if there's an amount on the same line after NON-TAXABLE
                same_line_match = re.search(r'NON-TAXABLE\s+(\d{1,3}(?:,\d{3})*\.\d{2})', line_clean, re.IGNORECASE)
                if same_line_match:
                    try:
                        amount = float(same_line_match.group(1).replace(',', ''))
                        if 100 <= amount <= 10000000:
                            return amount
                    except ValueError:
                        pass
                
                # Check next line for amount
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    amount_match = re.search(r'^(\d{1,3}(?:,\d{3})*\.\d{2})$', next_line)
                    if amount_match:
                        try:
                            amount = float(amount_match.group(1).replace(',', ''))
                            if 100 <= amount <= 10000000:
                                return amount
                        except ValueError:
                            continue
            
            # Pattern 4: Look for "AMOUNT CHARGED TO STORE ACCOUNT" pattern
            if 'AMOUNT CHARGED' in line_clean.upper() or 'STORE ACCOUNT' in line_clean.upper():
                # Look for amount in same line or next few lines
                for j in range(i, min(i + 3, len(lines))):
                    check_line = lines[j].strip()
                    amount_match = re.search(r'(\d{1,3}(?:,\d{3})*\.\d{2})', check_line)
                    if amount_match:
                        try:
                            amount = float(amount_match.group(1).replace(',', ''))
                            if 1000 <= amount <= 10000000:  # Higher minimum for this pattern
                                return amount
                        except ValueError:
                            continue
            
            # Pattern 5: FCVB specific - look for standalone amount after NON-TAXABLE line
            if i > 0 and 'NON-TAXABLE' in lines[i-1].upper():
                # Current line might be the amount
                amount_match = re.search(r'^(\d{1,3}(?:,\d{3})*\.\d{2})$', line_clean)
                if amount_match:
                    try:
                        amount = float(amount_match.group(1).replace(',', ''))
                        if 100 <= amount <= 10000000:
                            return amount
                    except ValueError:
                        continue
        
        return None

    def _find_amount_prefix(self, lines: List[str], current_index: int, partial_amount: str) -> Optional[float]:
        """
        Look for potential prefix digits in nearby lines to reconstruct split amounts.
        """
        
        # Check previous lines for potential prefixes
        for i in range(max(0, current_index - 3), current_index):
            line = lines[i].strip()
            
            # Look for lines with just digits that could be prefixes
            if re.match(r'^\s*\d{1,2}\s*$', line):  # 1-2 digits only
                prefix = line.strip()
                try:
                    # Try to reconstruct the amount
                    reconstructed_amount = float(prefix + partial_amount)
                    if 100 <= reconstructed_amount <= 10000000:  # Reasonable range
                        return reconstructed_amount
                except ValueError:
                    continue
        
        # Check next lines for potential prefixes (less common but possible)
        for i in range(current_index + 1, min(current_index + 3, len(lines))):
            line = lines[i].strip()
            
            if re.match(r'^\s*\d{1,2}\s*$', line):
                prefix = line.strip()
                try:
                    reconstructed_amount = float(prefix + partial_amount)
                    if 100 <= reconstructed_amount <= 10000000:
                        return reconstructed_amount
                except ValueError:
                    continue
        
        return None

    def _is_likely_false_positive(self, amount_str: str, full_text: str) -> bool:
        """Check if an amount is likely a false positive (zip code, phone, etc.)"""
        
        # Convert to number for checks
        try:
            amount = float(amount_str.replace(',', ''))
        except ValueError:
            return True
        
        # Common false positive patterns
        false_positive_indicators = [
            # Zip codes (5 digits, often starting with specific patterns)
            (len(amount_str.replace('.', '').replace(',', '')) == 5 and 
             amount_str.replace('.', '').replace(',', '').isdigit()),
            
            # Phone numbers (look for phone-like context)
            bool(re.search(rf'(?:PHONE|TEL|CALL|CONTACT)[\s\S]{{0,20}}{re.escape(amount_str)}', full_text, re.IGNORECASE)),
            
            # Dates (check if appears in date context)
            bool(re.search(rf'(?:DATE|DUE\s+DATE)[\s\S]{{0,20}}{re.escape(amount_str)}', full_text, re.IGNORECASE)),
            
            # Item/Part numbers (check if appears with item/part context)
            bool(re.search(rf'(?:ITEM|PART|SKU|MODEL)[\s\S]{{0,20}}{re.escape(amount_str)}', full_text, re.IGNORECASE)),
            
            # Very small amounts that are likely not invoice totals
            amount < 10,
            
            # Suspiciously round numbers that might be quantities
            (amount.is_integer() and amount < 100 and 
             bool(re.search(rf'{re.escape(amount_str)}[\s]*(?:PCS|PIECES|QTY|QUANTITY|UNITS?)', full_text, re.IGNORECASE))),
            
            # Amounts that appear in unit price context (per ton, per item, etc.)
            bool(re.search(rf'{re.escape(amount_str)}[\s]*(?:/TN|/UNIT|/PC|/EA|EACH)', full_text, re.IGNORECASE)),
        ]
        
        return any(false_positive_indicators)

    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract invoice date"""
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try different date formats
                    date_formats = [
                        '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
                        '%Y/%m/%d', '%Y-%m-%d', '%Y/%d/%m', '%Y-%d-%m',
                        '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y'
                    ]
                    
                    for fmt in date_formats:
                        try:
                            return datetime.strptime(match, fmt)
                        except ValueError:
                            continue
                            
                except Exception:
                    continue
        
        return None

    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number"""
        
        text_upper = text.upper()
        
        # Enhanced invoice number patterns with more specific context
        enhanced_invoice_patterns = [
            r'INVOICE\s+NUMBER[:\s]*([A-Z0-9\-/]+)',  # Invoice Number: 48465
            r'INV\s*#?[:\s]*([A-Z0-9\-/]+)',  # INV #: 6289/L
            r'INVOICE[:\s]*#?[:\s]*([A-Z0-9\-/]+)',  # Invoice: #453549
            r'Invoice[:\s]*#?[:\s]*([A-Z0-9\-/]+)',  # Invoice: #453549 (case sensitive)
            r'NO\.\s*([A-Z0-9\-/]+)',
            r'BILL\s*#?[:\s]*([A-Z0-9\-/]+)',
            r'REF\s*#?[:\s]*([A-Z0-9\-/]+)',
        ]
        
        # Try enhanced patterns first and collect all valid matches
        all_matches = []
        for pattern in enhanced_invoice_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                for match in matches:
                    match = match.strip()
                    # Skip if it's just numbers that could be dates or amounts
                    # Also skip common non-invoice-number words
                    if (match and 
                        not re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', match) and
                        match not in ['STORAGE', 'NUMBER', 'DATE', 'OICE', 'TOTAL']):
                        all_matches.append(match)
        
        # Prioritize matches from "Invoice Number:" context first, then by length
        if all_matches:
            # First, look for matches that came from "Invoice Number:" pattern
            invoice_number_matches = [match for match in all_matches if match.isdigit() and len(match) >= 4]
            if invoice_number_matches:
                return max(invoice_number_matches, key=len)
            else:
                return max(all_matches, key=len)
        
        # Try original patterns
        for pattern in self.invoice_number_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                for match in matches:
                    match = match.strip()
                    if (match and 
                        not re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', match) and
                        match not in ['STORAGE', 'NUMBER', 'DATE', 'OICE', 'TOTAL']):
                        return match
        
        return None

    def _extract_vendor(self, text: str, filename: str) -> str:
        """Extract vendor name"""
        
        # Enhanced vendor extraction patterns (prioritize company headers)
        enhanced_vendor_patterns = [
            r'^([A-Z][A-Z\s&,\.\-]*?WAREHOUSE)(?:\s|$)',  # FCVB WAREHOUSE (prioritize this)
            r'Smart Warehousing',  # Direct match for SWAK
            r'SMART WAREHOUSING',  # Direct match for SWAK (uppercase)
            r'REMIT\s+TO[:\s]*([A-Z][A-Z\s&,\.\-]+?)(?:\n|\r|P\.)',  # REMIT TO: FARMERS COOPERATIVE
            r'FROM[:\s]*([A-Z][A-Z\s&,\.\-]+?)(?:\n|\r|$)',
            r'BILL\s+FROM[:\s]*([A-Z][A-Z\s&,\.\-]+?)(?:\n|\r|$)',
            r'VENDOR[:\s]*([A-Z][A-Z\s&,\.\-]+?)(?:\n|\r|$)',
            r'COMPANY[:\s]*([A-Z][A-Z\s&,\.\-]+?)(?:\n|\r|$)',
        ]
        
        # Try enhanced patterns first
        for pattern in enhanced_vendor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                vendor = matches[0].strip()
                if len(vendor) > 3 and not re.match(r'^\d+$', vendor):  # Not just numbers
                    return vendor.title()
        
        # Try original patterns
        for pattern in self.vendor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                vendor = matches[0].strip()
                if len(vendor) > 3 and not re.match(r'^\d+$', vendor):
                    return vendor.title()
        
        # Smart filename parsing - avoid using invoice numbers as vendor names
        vendor_from_filename = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        
        # Remove copy indicators and file extensions
        vendor_from_filename = re.sub(r'\s*(copy|Copy)\s*\d*$', '', vendor_from_filename)
        vendor_from_filename = re.sub(r'\s*\d{4,}.*$', '', vendor_from_filename)  # Remove trailing numbers
        
        # Extract potential vendor name (letters only, not starting with numbers)
        vendor_match = re.match(r'^([A-Za-z][A-Za-z\s]+)', vendor_from_filename)
        if vendor_match:
            extracted_vendor = vendor_match.group(1).strip().title()
            # Don't use common non-vendor words
            if extracted_vendor not in ['Invoice', 'Bill', 'Receipt', 'Document', 'File']:
                return extracted_vendor
        
        return "Unknown Vendor"

    def process_multiple_pdfs(self, uploaded_files: List) -> pd.DataFrame:
        """Process multiple PDF files and return structured data"""
        
        processed_invoices = []
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Extract data
                invoice_data = self.extract_invoice_data(uploaded_file, uploaded_file.name)
                
                # Validate extracted data
                if not invoice_data['amount']:
                    errors.append(f"{uploaded_file.name}: Could not extract amount")
                    continue
                
                if not invoice_data['invoice_date']:
                    errors.append(f"{uploaded_file.name}: Could not extract date")
                    continue
                
                processed_invoices.append(invoice_data)
                
            except Exception as e:
                errors.append(f"{uploaded_file.name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show errors if any
        if errors:
            st.warning("Some files could not be processed:")
            for error in errors[:5]:  # Show first 5 errors
                st.text(f"• {error}")
            if len(errors) > 5:
                st.text(f"... and {len(errors) - 5} more errors")
        
        # Convert to DataFrame
        if processed_invoices:
            df = pd.DataFrame(processed_invoices)
            
            # Clean and standardize data
            df = self._clean_extracted_data(df)
            
            st.success(f"Successfully processed {len(df)} invoices from {len(uploaded_files)} files")
            return df
        else:
            raise ValueError("No invoices could be processed successfully")

    def _clean_extracted_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize extracted data"""
        
        # Ensure required columns exist
        required_columns = ['invoice_number', 'vendor', 'amount', 'invoice_date']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Clean amounts
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df[df['amount'] > 0].copy()  # Remove invalid amounts
        
        # Clean dates
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
        df = df[df['invoice_date'].notna()].copy()  # Remove invalid dates
        
        # Clean invoice numbers
        df['invoice_number'] = df['invoice_number'].fillna('UNKNOWN')
        df['invoice_number'] = df['invoice_number'].astype(str)
        
        # Clean vendor names
        df['vendor'] = df['vendor'].fillna('Unknown Vendor')
        df['vendor'] = df['vendor'].astype(str).str.strip().str.title()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['invoice_number', 'vendor', 'amount'], keep='first')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df

    def preview_extraction(self, pdf_file, filename: str) -> Dict:
        """Preview extraction for a single PDF (for testing)"""
        
        try:
            # Extract raw text
            text = self.extract_text_from_pdf(pdf_file)
            
            # Extract data
            invoice_data = self.extract_invoice_data(pdf_file, filename)
            
            return {
                'success': True,
                'extracted_data': invoice_data,
                'raw_text_sample': text[:1000] + "..." if len(text) > 1000 else text
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'raw_text_sample': ""
            }

    # Legacy methods for backward compatibility
    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file (legacy method for backward compatibility)"""
        try:
            with open(pdf_path, 'rb') as file:
                invoice_data = self.extract_invoice_data(file, pdf_path)
                return {
                    'file_path': pdf_path,
                    'amount': invoice_data['amount'],
                    'extracted_text': invoice_data['raw_text'],
                    'error': None if invoice_data['amount'] else 'No amount found'
                }
        except Exception as e:
            return {
                'file_path': pdf_path,
                'amount': None,
                'error': str(e)
            }

    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory (legacy method)"""
        directory = Path(directory_path)
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return []
        
        results = []
        for pdf_file in pdf_files:
            result = self.process_single_pdf(str(pdf_file))
            results.append(result)
        
        return results

    def export_results_to_csv(self, results: List[Dict[str, Any]], output_path: str = "invoice_results.csv"):
        """Export processing results to CSV (legacy method)"""
        df_data = []
        
        for result in results:
            df_data.append({
                'File_Path': result['file_path'],
                'Amount': result['amount'],
                'Error': result['error'],
                'Status': 'Success' if result['amount'] else 'Failed'
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = PDFInvoiceProcessor()
    
    # Process single PDF
    # result = processor.process_single_pdf("path/to/your/invoice.pdf")