import json
import os
import pandas as pd
from datetime import datetime
import requests
import subprocess

class InvoiceChatbot:
    """
    AI-powered chatbot for explaining invoice anomalies using local Llama models
    """
    
    def __init__(self, model_name="llama3.2:3b"):
        """
        Initialize the chatbot with local Llama model
        
        Args:
            model_name (str): Name of the Llama model to use ("llama3.2:3b" or "llama3.2:1b")
        """
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Test if Ollama is running
        if not self._test_ollama_connection():
            print("Warning: Ollama doesn't seem to be running. Please start it with 'ollama serve'")
    
    def _test_ollama_connection(self):
        """
        Test if Ollama is running and accessible
        
        Returns:
            bool: True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _call_llama(self, prompt, max_tokens=800, temperature=0.3):
        """
        Call local Llama model through Ollama
        
        Args:
            prompt (str): The prompt to send to Llama
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for response generation
            
        Returns:
            str: Generated response from Llama
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Sorry, I couldn't generate a response.")
            else:
                return f"Error: Ollama returned status code {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running with 'ollama serve'"
        except Exception as e:
            return f"Error calling Llama model: {str(e)}"
    
    def explain_anomaly(self, user_query, df, anomalies):
        """
        Generate AI explanation for invoice anomalies based on user query
        
        Args:
            user_query (str): User's question about anomalies
            df (pd.DataFrame): Invoice dataframe
            anomalies (dict): Dictionary of anomaly indices
            
        Returns:
            str: AI-generated explanation
        """
        try:
            # Prepare context about the anomalies
            context = self._prepare_anomaly_context(df, anomalies)
            
            # Create the full prompt for Llama
            full_prompt = f"""You are an expert invoice analyst specializing in 3PL (third-party logistics) operations. 
You help users understand invoice anomalies and their potential business impact.

Current anomaly analysis context:
{context}

Guidelines for responses:
1. Be clear and professional
2. Explain technical concepts in business terms
3. Suggest actionable next steps when appropriate
4. Reference specific data points when relevant
5. Consider potential business impact and risks
6. Be concise but thorough

User Question: {user_query}

Please provide a helpful response based on the anomaly data above:"""
            
            # Generate response using local Llama
            response = self._call_llama(full_prompt, max_tokens=800, temperature=0.3)
            return response
            
        except Exception as e:
            return f"I apologize, but I'm currently unable to process your request. Error: {str(e)}"
    
    def _prepare_anomaly_context(self, df, anomalies):
        """
        Prepare contextual information about anomalies for the AI
        
        Args:
            df (pd.DataFrame): Invoice dataframe
            anomalies (dict): Dictionary of anomaly indices
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        # Basic dataset information
        context_parts.append(f"Dataset Overview:")
        context_parts.append(f"- Total invoices: {len(df):,}")
        context_parts.append(f"- Date range: {df['invoice_date'].min().strftime('%Y-%m-%d')} to {df['invoice_date'].max().strftime('%Y-%m-%d')}")
        context_parts.append(f"- Unique vendors: {df['vendor'].nunique()}")
        context_parts.append(f"- Total invoice value: ${df['amount'].sum():,.2f}")
        context_parts.append("")
        
        # Anomaly summary
        context_parts.append("Anomaly Summary:")
        total_anomalies = sum(len(indices) for indices in anomalies.values())
        context_parts.append(f"- Total anomalies detected: {total_anomalies}")
        
        for anomaly_type, indices in anomalies.items():
            if indices:
                anomalous_invoices = df.loc[indices]
                percentage = (len(indices) / len(df)) * 100
                total_amount = anomalous_invoices['amount'].sum()
                
                context_parts.append(f"- {anomaly_type.replace('_', ' ').title()}: {len(indices)} invoices ({percentage:.1f}%), ${total_amount:,.2f}")
        
        context_parts.append("")
        
        # Top anomalous invoices by amount
        if total_anomalies > 0:
            all_anomaly_indices = []
            for indices in anomalies.values():
                all_anomaly_indices.extend(indices)
            
            unique_anomaly_indices = list(set(all_anomaly_indices))
            top_anomalies = df.loc[unique_anomaly_indices].nlargest(5, 'amount')
            
            context_parts.append("Top 5 Anomalous Invoices by Amount:")
            for idx, row in top_anomalies.iterrows():
                anomaly_types = []
                for anom_type, indices in anomalies.items():
                    if idx in indices:
                        anomaly_types.append(anom_type.replace('_', ' '))
                
                context_parts.append(f"- Invoice {row['invoice_number']}: ${row['amount']:,.2f} from {row['vendor']} ({', '.join(anomaly_types)})")
        
        context_parts.append("")
        
        # Vendor analysis
        vendor_anomaly_counts = {}
        for anomaly_type, indices in anomalies.items():
            if indices:
                vendor_counts = df.loc[indices]['vendor'].value_counts()
                for vendor, count in vendor_counts.items():
                    if vendor not in vendor_anomaly_counts:
                        vendor_anomaly_counts[vendor] = {}
                    vendor_anomaly_counts[vendor][anomaly_type] = count
        
        if vendor_anomaly_counts:
            context_parts.append("Vendors with Most Anomalies:")
            sorted_vendors = sorted(vendor_anomaly_counts.items(), 
                                  key=lambda x: sum(x[1].values()), reverse=True)[:5]
            
            for vendor, anomaly_counts in sorted_vendors:
                total_vendor_anomalies = sum(anomaly_counts.values())
                context_parts.append(f"- {vendor}: {total_vendor_anomalies} anomalies")
        
        return "\n".join(context_parts)
    
    def get_specific_invoice_analysis(self, df, invoice_number):
        """
        Get detailed analysis for a specific invoice
        
        Args:
            df (pd.DataFrame): Invoice dataframe
            invoice_number (str): Invoice number to analyze
            
        Returns:
            str: Detailed invoice analysis
        """
        try:
            # Find the invoice
            invoice_data = df[df['invoice_number'] == invoice_number]
            
            if invoice_data.empty:
                return f"Invoice {invoice_number} not found in the dataset."
            
            if len(invoice_data) > 1:
                return f"Multiple invoices found with number {invoice_number}. Please be more specific."
            
            invoice = invoice_data.iloc[0]
            
            # Prepare detailed analysis
            analysis_prompt = f"""You are an expert invoice analyst. Provide detailed analysis of this specific invoice, including any potential concerns or notable patterns.

Analyze this specific invoice in detail:

Invoice Details:
- Invoice Number: {invoice['invoice_number']}
- Vendor: {invoice['vendor']}
- Amount: ${invoice['amount']:,.2f}
- Date: {invoice['invoice_date'].strftime('%Y-%m-%d')}

Context from similar invoices:
"""
            
            # Add context from same vendor
            vendor_invoices = df[df['vendor'] == invoice['vendor']]
            if len(vendor_invoices) > 1:
                vendor_stats = {
                    'count': len(vendor_invoices),
                    'avg_amount': vendor_invoices['amount'].mean(),
                    'total_amount': vendor_invoices['amount'].sum(),
                    'date_range': f"{vendor_invoices['invoice_date'].min().strftime('%Y-%m-%d')} to {vendor_invoices['invoice_date'].max().strftime('%Y-%m-%d')}"
                }
                
                analysis_prompt += f"""
Same Vendor ({invoice['vendor']}) Statistics:
- Total invoices: {vendor_stats['count']}
- Average amount: ${vendor_stats['avg_amount']:,.2f}
- Total value: ${vendor_stats['total_amount']:,.2f}
- Date range: {vendor_stats['date_range']}

This invoice comparison:
- Amount vs vendor average: {((invoice['amount'] / vendor_stats['avg_amount'] - 1) * 100):+.1f}%

Please provide a detailed analysis of this invoice:"""
            
            response = self._call_llama(analysis_prompt, max_tokens=600, temperature=0.3)
            return response
            
        except Exception as e:
            return f"Error analyzing invoice: {str(e)}"
    
    def suggest_actions(self, df, anomalies):
        """
        Suggest actionable steps based on detected anomalies
        
        Args:
            df (pd.DataFrame): Invoice dataframe
            anomalies (dict): Dictionary of anomaly indices
            
        Returns:
            str: AI-generated action recommendations
        """
        try:
            # Prepare summary for action suggestions
            summary = {
                'total_invoices': len(df),
                'total_anomalies': sum(len(indices) for indices in anomalies.values()),
                'financial_impact': sum(df.loc[indices]['amount'].sum() for indices in anomalies.values() if indices),
                'anomaly_breakdown': {k: len(v) for k, v in anomalies.items()}
            }
            
            action_prompt = f"""You are a business process consultant specializing in accounts payable and invoice management. Provide practical, actionable recommendations.

Based on the following invoice anomaly analysis, provide specific, actionable recommendations:

Summary:
- Total invoices analyzed: {summary['total_invoices']:,}
- Total anomalies detected: {summary['total_anomalies']}
- Financial impact: ${summary['financial_impact']:,.2f}
- Anomaly breakdown: {summary['anomaly_breakdown']}

Please provide:
1. Immediate actions to take
2. Process improvements to prevent future anomalies
3. Risk mitigation strategies
4. Monitoring recommendations

Your response:"""
            
            response = self._call_llama(action_prompt, max_tokens=800, temperature=0.3)
            return response
            
        except Exception as e:
            return f"Error generating action suggestions: {str(e)}"

    def change_model(self, new_model_name):
        """
        Change the Llama model being used
        
        Args:
            new_model_name (str): New model name ("llama3.2:3b" or "llama3.2:1b")
        """
        self.model_name = new_model_name
        print(f"Switched to model: {new_model_name}")