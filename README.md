# 3PL Invoice Anomaly Detection System

A comprehensive Python-based system for detecting anomalies in 3rd Party Logistics (3PL) invoices using statistical analysis, duplicate detection, and AI-powered explanations.

## Features

### Core Functionality
- **PDF Invoice Processing**: Automatically extract data from PDF invoices
- **Multi-format Support**: Process PDF, CSV, and Excel files
- **Statistical Anomaly Detection**: Detect outliers using 3-sigma rule and rolling window analysis
- **Duplicate Detection**: Identify potential duplicate invoices using multiple criteria
- **Future Date Detection**: Flag invoices with future dates
- **AI Chatbot**: Get detailed explanations of anomalies using llama 3b - Can configure any model

### Anomaly Detection Methods

#### 1. Future-Dated Invoices
- Detects invoices with dates in the future
- Flags potential data entry errors

#### 2. Duplicate Invoice Detection
- **Exact Duplicates**: Same invoice number, vendor, and amount
- **Suspicious Duplicates**: Same invoice number/vendor with different amounts
- **Amount/Date Duplicates**: Same vendor, amount, and date with different invoice numbers
- **Similar Amount Detection**: Amounts within 1% on the same day from same vendor

#### 3. Statistical Anomaly Detection
- **Rolling Window Analysis**: Uses configurable months (default 9) for mean calculation
- **3-Sigma Rule**: Configurable threshold (default 3.0 standard deviations)
- **Vendor-Specific Analysis**: Individual statistical analysis per vendor
- **Global Analysis**: Cross-vendor outlier detection using multiple methods:
  - Tukey's method (IQR-based outliers)
  - Modified Z-score using Median Absolute Deviation
  - Extreme outlier detection (5+ standard deviations)

## Installation

### Prerequisites
- Python 3.11+
- Required packages (automatically installed):
  - streamlit
  - pandas
  - numpy
  - plotly
  - pdfplumber
  - PyPDF2
  - python-multipart
  - scipy
  - openai
  - openpyxl

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set OpenAI API key (optional, for AI chatbot feature)
4. Run the application: `streamlit run app.py --server.port 5000`

## File Structure

```
├── app.py                 # Main Streamlit application
├── invoice_analyzer.py    # Data loading and preprocessing
├── pdf_processor.py       # PDF text extraction and data parsing
├── anomaly_detector.py    # Anomaly detection algorithms
├── chatbot.py            # AI-powered anomaly explanations
├── README.md             # This documentation
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## Usage

### 1. Upload Data
- Navigate to the "Upload Data" tab
- Upload PDF invoices, CSV files, or Excel files
- The system automatically processes different file types
- Configure rolling window (3-24 months) and sigma threshold (1.0-5.0)

### 2. View Analysis Results
- Check the "Analysis Results" tab after processing
- View summary metrics and detailed anomaly breakdowns
- Interactive charts show amount distributions and timeline views

### 3. AI Assistant
- Use the "AI Assistant" tab for detailed explanations
- Ask questions about specific anomalies
- Get actionable recommendations

### 4. Export Results
- Download CSV or Excel reports
- Includes detailed anomaly descriptions
- Summary statistics included

## API Reference

### InvoiceAnalyzer Class
```python
analyzer = InvoiceAnalyzer()
df = analyzer.load_data(uploaded_files)  # Supports PDF, CSV, Excel
```

### AnomalyDetector Class
```python
detector = AnomalyDetector(rolling_months=9, sigma_threshold=3.0)
anomalies = detector.detect_all_anomalies(df)
```

### PDFInvoiceProcessor Class
```python
processor = PDFInvoiceProcessor()
data = processor.extract_invoice_data(pdf_file, filename)
```

### InvoiceChatbot Class
```python
chatbot = InvoiceChatbot()
response = chatbot.explain_anomaly(query, df, anomalies)
```

## Configuration

### Anomaly Detection Parameters
- **Rolling Months**: 3-24 months (default: 9)
- **Sigma Threshold**: 1.0-5.0 (default: 3.0)

### PDF Processing
- Supports multiple extraction methods (pdfplumber, PyPDF2)
- Automatic fallback for difficult PDFs
- Configurable regex patterns for data extraction

## Data Requirements

### For CSV/Excel Files
Required columns:
- `invoice_number`: Unique invoice identifier
- `amount`: Invoice amount (numeric)
- `vendor`: Vendor/supplier name
- `invoice_date`: Invoice date

### For PDF Files
The system attempts to extract:
- Invoice numbers (various formats)
- Amounts (with currency symbols)
- Dates (multiple date formats)
- Vendor names (from text or filename)

## Anomaly Analysis Details

### Statistical Methods Used

1. **Rolling Window Statistics**
   - Calculates moving averages and standard deviations
   - Vendor-specific analysis for better accuracy
   - Configurable time window (default: 9 months)

2. **Outlier Detection Algorithms**
   - Z-score analysis (3-sigma rule)
   - Tukey's method (Interquartile Range)
   - Modified Z-score (Median Absolute Deviation)
   - Isolation Forest principles

3. **Duplicate Detection Logic**
   - Multiple similarity criteria
   - Fuzzy matching for amounts (1% tolerance)
   - Temporal clustering analysis

### Risk Assessment

Each anomaly type has different risk levels:
- **High Risk**: Future-dated invoices, exact duplicates
- **Medium Risk**: Statistical outliers, suspicious duplicates
- **Low Risk**: Vendor frequency anomalies

## AI Chatbot Features

- Powered by OpenAI GPT-4
- Context-aware responses
- Specific invoice analysis
- Actionable recommendations
- Business impact assessment

## Troubleshooting

### Common Issues
1. **PDF Processing Errors**: Check PDF quality and text readability
2. **Missing Data**: Ensure required columns exist in CSV/Excel files
3. **Low Extraction Accuracy**: PDFs with images/scanned text may need OCR

### Performance Tips
- Process PDFs in smaller batches for better performance
- Use CSV/Excel for large datasets when possible
- Configure appropriate rolling window based on data volume

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - see LICENSE file for details
