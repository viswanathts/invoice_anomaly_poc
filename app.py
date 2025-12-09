import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json

from invoice_analyzer import InvoiceAnalyzer
from anomaly_detector import AnomalyDetector

from chatbot import InvoiceChatbot

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="3PL Invoice Anomaly Detection",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔍 3PL Invoice Anomaly Detection System")
    st.markdown("Upload your invoice data to detect anomalies, duplicates, and get AI-powered explanations.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Rolling window configuration
        rolling_months = st.slider(
            "Rolling Window (months)",
            min_value=3,
            max_value=24,
            value=9,
            help="Number of months to use for statistical analysis"
        )
        
        # Sigma threshold
        sigma_threshold = st.slider(
            "Sigma Threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Standard deviation threshold for anomaly detection"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system detects:
        - **Future-dated invoices**
        - **Duplicate invoices**
        - **Statistical anomalies**
        - **Amount outliers**
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "📊 Analysis Results", "🤖 AI Assistant", "📋 Export"])
    
    with tab1:
        upload_data_tab(rolling_months, sigma_threshold)
    
    with tab2:
        analysis_results_tab()
    
    with tab3:
        ai_assistant_tab()
    
    with tab4:
        export_tab()

def upload_data_tab(rolling_months, sigma_threshold):
    st.header("Upload Invoice Data")
    
    uploaded_files = st.file_uploader(
        "Choose PDF, CSV, or Excel files",
        type=['pdf', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload your 3PL invoice files (PDF invoices or CSV/Excel data)"
    )
    
    if uploaded_files:
        try:
            # Load data
            analyzer = InvoiceAnalyzer()
            df = analyzer.load_data(uploaded_files)
            
            st.success(f"✅ Successfully loaded {len(df)} invoices")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Invoices", len(df))
            with col2:
                st.metric("Date Range", f"{df['invoice_date'].min().date()} to {df['invoice_date'].max().date()}")
            with col3:
                st.metric("Unique Vendors", int(df['vendor'].nunique()))
            with col4:
                st.metric("Total Amount", f"${df['amount'].sum():,.2f}")
            
            # Process anomalies
            if st.button("🔍 Analyze for Anomalies", type="primary"):
                with st.spinner("Analyzing invoices for anomalies..."):
                    detector = AnomalyDetector(rolling_months=rolling_months, sigma_threshold=sigma_threshold)
                    anomalies = detector.detect_all_anomalies(df)
                    
                    st.session_state.uploaded_data = df
                    st.session_state.anomalies = anomalies
                    
                    st.success("✅ Analysis complete! Check the Analysis Results tab.")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your files are valid PDFs with invoice data, or CSV/Excel files with columns: invoice_number, amount, vendor, invoice_date")

def analysis_results_tab():
    if st.session_state.anomalies is None:
        st.info("📤 Please upload and analyze data first in the Upload Data tab.")
        return
    
    anomalies = st.session_state.anomalies
    df = st.session_state.uploaded_data
    
    st.header("Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Future Dated", len(anomalies['future_dated']))
    with col2:
        st.metric("Duplicates", len(anomalies['duplicates']))
    with col3:
        st.metric("Statistical Anomalies", len(anomalies['statistical']))
    with col4:
        total_anomalies = len(anomalies['future_dated']) + len(anomalies['duplicates']) + len(anomalies['statistical'])
        st.metric("Total Anomalies", total_anomalies)
    
    # Anomaly breakdown
    if total_anomalies > 0:
        # Future dated invoices
        if len(anomalies['future_dated']) > 0:
            st.subheader("🔮 Future-Dated Invoices")
            future_df = df[df.index.isin(anomalies['future_dated'])]
            st.dataframe(future_df[['invoice_number', 'vendor', 'amount', 'invoice_date']], use_container_width=True)
        
        # Duplicate invoices
        if len(anomalies['duplicates']) > 0:
            st.subheader("👥 Duplicate Invoices")
            duplicate_df = df[df.index.isin(anomalies['duplicates'])]
            st.dataframe(duplicate_df[['invoice_number', 'vendor', 'amount', 'invoice_date']], use_container_width=True)
        
        # Statistical anomalies
        if len(anomalies['statistical']) > 0:
            st.subheader("📈 Statistical Anomalies")
            statistical_df = df[df.index.isin(anomalies['statistical'])]
            st.dataframe(statistical_df[['invoice_number', 'vendor', 'amount', 'invoice_date']], use_container_width=True)
            
            # Amount distribution chart
            fig = px.histogram(df, x='amount', nbins=50, title='Invoice Amount Distribution')
            
            # Highlight anomalies
            anomaly_amounts = statistical_df['amount'].tolist()
            fig.add_vline(x=df['amount'].mean(), line_dash="dash", line_color="green", 
                         annotation_text="Mean")
            
            for amount in anomaly_amounts:
                fig.add_vline(x=amount, line_color="red", opacity=0.3)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline visualization
        st.subheader("📅 Anomaly Timeline")
        
        # Prepare timeline data
        timeline_data = []
        
        for idx in anomalies['future_dated']:
            row = df.loc[idx]
            timeline_data.append({
                'date': row['invoice_date'],
                'type': 'Future Dated',
                'invoice': row['invoice_number'],
                'amount': row['amount'],
                'vendor': row['vendor']
            })
        
        for idx in anomalies['duplicates']:
            row = df.loc[idx]
            timeline_data.append({
                'date': row['invoice_date'],
                'type': 'Duplicate',
                'invoice': row['invoice_number'],
                'amount': row['amount'],
                'vendor': row['vendor']
            })
        
        for idx in anomalies['statistical']:
            row = df.loc[idx]
            timeline_data.append({
                'date': row['invoice_date'],
                'type': 'Statistical',
                'invoice': row['invoice_number'],
                'amount': row['amount'],
                'vendor': row['vendor']
            })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            fig = px.scatter(timeline_df, x='date', y='amount', color='type',
                           hover_data=['invoice', 'vendor'],
                           title='Anomalies Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.success("🎉 No anomalies detected in your invoice data!")

def ai_assistant_tab():
    if st.session_state.anomalies is None:
        st.info("📤 Please upload and analyze data first to use the AI assistant.")
        return
    
    st.header("🤖 AI Assistant")
    st.markdown("Ask questions about specific invoice anomalies and get detailed explanations.")
    
    # Initialize chatbot
    chatbot = InvoiceChatbot()
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about invoice anomalies..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chatbot.explain_anomaly(
                        prompt, 
                        st.session_state.uploaded_data, 
                        st.session_state.anomalies
                    )
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def export_tab():
    if st.session_state.anomalies is None:
        st.info("📤 Please upload and analyze data first to export results.")
        return
    
    st.header("📋 Export Results")
    
    anomalies = st.session_state.anomalies
    df = st.session_state.uploaded_data
    
    # Prepare export data
    export_data = []
    
    # Future dated
    for idx in anomalies['future_dated']:
        row = df.loc[idx]
        export_data.append({
            'invoice_number': row['invoice_number'],
            'vendor': row['vendor'],
            'amount': row['amount'],
            'invoice_date': row['invoice_date'],
            'anomaly_type': 'Future Dated',
            'description': f'Invoice date {row["invoice_date"].date()} is in the future'
        })
    
    # Duplicates
    for idx in anomalies['duplicates']:
        row = df.loc[idx]
        export_data.append({
            'invoice_number': row['invoice_number'],
            'vendor': row['vendor'],
            'amount': row['amount'],
            'invoice_date': row['invoice_date'],
            'anomaly_type': 'Duplicate',
            'description': 'Potential duplicate invoice based on multiple criteria'
        })
    
    # Statistical
    for idx in anomalies['statistical']:
        row = df.loc[idx]
        export_data.append({
            'invoice_number': row['invoice_number'],
            'vendor': row['vendor'],
            'amount': row['amount'],
            'invoice_date': row['invoice_date'],
            'anomaly_type': 'Statistical Anomaly',
            'description': f'Amount ${row["amount"]:,.2f} is a statistical outlier'
        })
    
    if export_data:
        export_df = pd.DataFrame(export_data)
        
        st.subheader("Anomaly Report Preview")
        st.dataframe(export_df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Report",
                data=csv,
                file_name=f"invoice_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl', mode='w') as writer:
                export_df.to_excel(writer, sheet_name='Anomalies', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Anomaly Type': ['Future Dated', 'Duplicates', 'Statistical'],
                    'Count': [len(anomalies['future_dated']), len(anomalies['duplicates']), len(anomalies['statistical'])]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                label="📊 Download Excel Report",
                data=buffer.getvalue(),
                file_name=f"invoice_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("No anomalies to export.")

if __name__ == "__main__":
    main()
