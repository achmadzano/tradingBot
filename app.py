import streamlit as st
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from langchain_sambanova import ChatSambaNovaCloud
from langchain.schema import HumanMessage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# Load environment variables
load_dotenv()

def parse_trading_signal(signal_text):
    """Parse AI trading signal response into structured data"""
    try:
        # Extract key information using regex
        signal_pattern = r"TRADING SIGNAL:\s*([A-Z]+)"
        entry_pattern = r"ENTRY PRICE:\s*([\d.]+)"
        tp_pattern = r"TAKE PROFIT:\s*([\d.]+)"
        sl_pattern = r"STOP LOSS:\s*([\d.]+)"
        confidence_pattern = r"CONFIDENCE:\s*([A-Za-z]+)"
        reasoning_pattern = r"REASONING:\s*(.+?)(?=\n\n|$)"
        
        parsed = {
            'signal': re.search(signal_pattern, signal_text, re.IGNORECASE),
            'entry_price': re.search(entry_pattern, signal_text),
            'take_profit': re.search(tp_pattern, signal_text),
            'stop_loss': re.search(sl_pattern, signal_text),
            'confidence': re.search(confidence_pattern, signal_text, re.IGNORECASE),
            'reasoning': re.search(reasoning_pattern, signal_text, re.DOTALL | re.IGNORECASE)
        }
        
        # Extract values
        result = {}
        for key, match in parsed.items():
            if match:
                result[key] = match.group(1).strip()
            else:
                result[key] = "N/A"
        
        return result
    except Exception as e:
        return {
            'signal': 'N/A',
            'entry_price': 'N/A', 
            'take_profit': 'N/A',
            'stop_loss': 'N/A',
            'confidence': 'N/A',
            'reasoning': signal_text
        }

def display_enhanced_signal(parsed_signal, symbol):
    """Display trading signal with enhanced visual formatting"""
    
    # Get signal type and set colors
    signal_type = parsed_signal.get('signal', 'N/A').upper()
    
    if signal_type == 'LONG':
        signal_color = "🟢"
        signal_emoji = "📈"
        bg_color = "#d4edda"
        border_color = "#28a745"
    elif signal_type == 'SHORT':
        signal_color = "🔴"
        signal_emoji = "📉"
        bg_color = "#f8d7da"
        border_color = "#dc3545"
    else:
        signal_color = "🟡"
        signal_emoji = "⏸️"
        bg_color = "#fff3cd"
        border_color = "#ffc107"
    
    # Main signal display - more compact
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border: 2px solid {border_color};
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
    ">
        <h3 style="margin: 0; color: {border_color}; font-size: 1.2em;">
            {signal_emoji} {signal_type} SIGNAL {signal_color}
        </h3>
        <p style="margin: 3px 0; font-size: 0.9em; color: #666;">
            {symbol}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Price levels in columns - more compact
    col1, col2, col3 = st.columns(3)
    
    with col1:
        entry_price = parsed_signal.get('entry_price', 'N/A')
        st.markdown(f"""
        <div style="
            background-color: #e3f2fd;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            border-left: 3px solid #2196f3;
            margin: 5px 0;
        ">
            <p style="margin: 0; color: #1976d2; font-size: 0.8em; font-weight: bold;">🎯 Entry Price</p>
            <p style="margin: 3px 0; color: #0d47a1; font-size: 1em; font-weight: bold;">{entry_price}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tp_price = parsed_signal.get('take_profit', 'N/A')
        st.markdown(f"""
        <div style="
            background-color: #e8f5e8;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            border-left: 3px solid #4caf50;
            margin: 5px 0;
        ">
            <p style="margin: 0; color: #388e3c; font-size: 0.8em; font-weight: bold;">💰 Take Profit</p>
            <p style="margin: 3px 0; color: #1b5e20; font-size: 1em; font-weight: bold;">{tp_price}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sl_price = parsed_signal.get('stop_loss', 'N/A')
        st.markdown(f"""
        <div style="
            background-color: #ffebee;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            border-left: 3px solid #f44336;
            margin: 5px 0;
        ">
            <p style="margin: 0; color: #d32f2f; font-size: 0.8em; font-weight: bold;">🛡️ Stop Loss</p>
            <p style="margin: 3px 0; color: #b71c1c; font-size: 1em; font-weight: bold;">{sl_price}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence and Risk-Reward - more compact
    col4, col5 = st.columns(2)
    
    with col4:
        confidence = parsed_signal.get('confidence', 'N/A').upper()
        if confidence == 'HIGH':
            conf_color = "#4caf50"
            conf_emoji = "🟢"
        elif confidence == 'MEDIUM':
            conf_color = "#ff9800"
            conf_emoji = "🟡"
        else:
            conf_color = "#f44336"
            conf_emoji = "🔴"
        
        st.markdown(f"""
        <div style="
            background-color: #f9f9f9;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
            border: 1px solid {conf_color};
            margin: 5px 0;
        ">
            <p style="margin: 0; color: {conf_color}; font-size: 0.8em; font-weight: bold;">{conf_emoji} Confidence</p>
            <p style="margin: 3px 0; color: {conf_color}; font-size: 0.9em; font-weight: bold;">{confidence}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        # Calculate Risk-Reward ratio
        try:
            entry = float(parsed_signal.get('entry_price', 0))
            tp = float(parsed_signal.get('take_profit', 0))
            sl = float(parsed_signal.get('stop_loss', 0))
            
            if entry > 0 and tp > 0 and sl > 0:
                if signal_type == 'LONG':
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                else:  # SHORT
                    risk = abs(sl - entry)
                    reward = abs(entry - tp)
                
                if risk > 0:
                    rr_ratio = round(reward / risk, 2)
                    rr_text = f"1:{rr_ratio}"
                    rr_color = "#4caf50" if rr_ratio >= 2 else "#ff9800" if rr_ratio >= 1.5 else "#f44336"
                else:
                    rr_text = "N/A"
                    rr_color = "#666"
            else:
                rr_text = "N/A"
                rr_color = "#666"
        except:
            rr_text = "N/A"
            rr_color = "#666"
        
        st.markdown(f"""
        <div style="
            background-color: #f9f9f9;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
            border: 1px solid {rr_color};
            margin: 5px 0;
        ">
            <p style="margin: 0; color: {rr_color}; font-size: 0.8em; font-weight: bold;">⚖️ Risk:Reward</p>
            <p style="margin: 3px 0; color: {rr_color}; font-size: 0.9em; font-weight: bold;">{rr_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Reasoning - more compact
    reasoning = parsed_signal.get('reasoning', 'No reasoning provided.')
    st.markdown(f"""
    <div style="
        background-color: #fafafa;
        border-radius: 6px;
        padding: 12px;
        margin: 10px 0;
        border-left: 3px solid #673ab7;
    ">
        <p style="margin: 0 0 5px 0; color: #673ab7; font-size: 0.9em; font-weight: bold;">🧠 AI Analysis & Reasoning</p>
        <p style="margin: 0; line-height: 1.4; color: #333; font-size: 0.85em;">{reasoning}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trading action buttons (visual only) - smaller buttons
    # st.markdown("#### 🎯 Quick Actions")
    # action_col1, action_col2, action_col3 = st.columns(3)
    
    # with action_col1:
    #     if signal_type in ['LONG', 'SHORT']:
    #         if st.button(f"📋 Copy {signal_type}", use_container_width=True, key="copy_signal"):
    #             st.success("Signal copied!")
    
    # with action_col2:
    #     if st.button("📊 Price Alert", use_container_width=True, key="price_alert"):
    #         st.info("Alert feature coming soon!")
    
    # with action_col3:
    #     if st.button("📝 Watchlist", use_container_width=True, key="watchlist"):
    #         st.info("Watchlist feature coming soon!")

# Trading Bot class remains unchanged
class TradingBot:
    def __init__(self):
        self.sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
        self.selected_model = "Llama-4-Maverick-17B-128E-Instruct"  # Fixed model
        self.setup_llm()
    
    def setup_llm(self):
        """Initialize Llama LLM through LangChain"""
        try:
            if self.sambanova_api_key:
                self.llm = ChatSambaNovaCloud(
                    model=self.selected_model,
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.01,
                )
            else:
                self.llm = None
        except Exception as e:
            st.error(f"Error initializing Llama LLM: {e}")
            self.llm = None
    
    def get_market_data(self, symbol, period="5d", interval="5m"):
        """Fetch market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        if data is None or len(data) < 50:
            return data
        
        # Calculate moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_9'] = data['Close'].ewm(span=9).mean()
        data['EMA_21'] = data['Close'].ewm(span=21).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = ((data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']) * 100
        data['BB_Position'] = ((data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])) * 100
        
        # Calculate Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stoch_K'] = ((data['Close'] - low_14) / (high_14 - low_14)) * 100
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        
        # Calculate Williams %R
        data['Williams_R'] = ((high_14 - data['Close']) / (high_14 - low_14)) * -100
        
        # Calculate ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['ATR'] = true_range.rolling(window=14).mean()
        
        # Calculate Commodity Channel Index (CCI)
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        data['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Calculate ADX (Average Directional Index)
        plus_dm = data['High'].diff()
        minus_dm = data['Low'].diff() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr_14 = true_range.rolling(window=14).sum()
        plus_di_14 = 100 * (plus_dm.rolling(window=14).sum() / tr_14)
        minus_di_14 = 100 * (minus_dm.rolling(window=14).sum() / tr_14)
        
        dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
        data['ADX'] = dx.rolling(window=14).mean()
        data['Plus_DI'] = plus_di_14
        data['Minus_DI'] = minus_di_14
        
        # Calculate Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # OBV (On Balance Volume)
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['OBV'] = obv
        
        # Price patterns and momentum
        data['Price_Change_Pct'] = data['Close'].pct_change() * 100
        data['Volatility'] = data['Price_Change_Pct'].rolling(window=20).std()
        
        # Fibonacci retracement levels (dynamic)
        recent_high = data['High'].rolling(window=50).max()
        recent_low = data['Low'].rolling(window=50).min()
        fib_range = recent_high - recent_low
        
        data['Fib_23.6'] = recent_high - (fib_range * 0.236)
        data['Fib_38.2'] = recent_high - (fib_range * 0.382)
        data['Fib_50.0'] = recent_high - (fib_range * 0.5)
        data['Fib_61.8'] = recent_high - (fib_range * 0.618)
        
        return data
    
    def analyze_market_structure(self, data):
        """Analyze market structure for support/resistance levels"""
        if data is None or len(data) < 10:
            return {}
        
        recent_data = data.tail(50)
        current_price = recent_data['Close'].iloc[-1]
        
        # Find recent highs and lows
        highs = recent_data['High'].nlargest(5)
        lows = recent_data['Low'].nsmallest(5)
        
        # Calculate support and resistance levels
        resistance_levels = highs[highs > current_price].tolist()
        support_levels = lows[lows < current_price].tolist()
        
        return {
            'current_price': current_price,
            'resistance_levels': resistance_levels[:3],  # Top 3 resistance levels
            'support_levels': support_levels[:3],  # Top 3 support levels
            'recent_high': recent_data['High'].max(),
            'recent_low': recent_data['Low'].min(),
        }
    
    def create_market_analysis_prompt(self, symbol, data, market_structure, interval="5m"):
        """Create a comprehensive prompt for LLM analysis"""
        if data is None or len(data) < 5:
            return None
        
        recent_data = data.tail(10)
        current_candle = data.iloc[-1]
        prev_candle = data.iloc[-2]
        
        # Current market conditions
        current_price = current_candle['Close']
        price_change = ((current_price - prev_candle['Close']) / prev_candle['Close']) * 100
        
        # Technical indicators - format them properly before using in f-string
        rsi = current_candle.get('RSI', 'N/A')
        rsi_formatted = f"{rsi:.2f}" if isinstance(rsi, (int, float)) and not pd.isna(rsi) else "N/A"
        
        macd = current_candle.get('MACD', 'N/A')
        macd_formatted = f"{macd:.5f}" if isinstance(macd, (int, float)) and not pd.isna(macd) else "N/A"
        
        macd_signal = current_candle.get('MACD_Signal', 'N/A')
        macd_signal_formatted = f"{macd_signal:.5f}" if isinstance(macd_signal, (int, float)) and not pd.isna(macd_signal) else "N/A"
        
        prompt = f"""
        You are an expert forex/crypto/commodities trading analyst. Analyze the following market data for {symbol} and provide a specific trading recommendation.

        CURRENT MARKET CONDITIONS:
        - Symbol: {symbol}
        - Current Price: {current_price:.5f}
        - Price Change (last candle): {price_change:.2f}%
        - Timeframe: {interval} chart
        
        TECHNICAL INDICATORS:
        - RSI (14): {rsi_formatted}
        - MACD: {macd_formatted}
        - MACD Signal: {macd_signal_formatted}
        
        MARKET STRUCTURE:
        - Recent High: {market_structure.get('recent_high', 'N/A')}
        - Recent Low: {market_structure.get('recent_low', 'N/A')}
        - Key Resistance Levels: {market_structure.get('resistance_levels', [])}
        - Key Support Levels: {market_structure.get('support_levels', [])}
        
        RECENT PRICE ACTION (Last 5 candles):
        {self.format_recent_candles(recent_data.tail(5))}
        
        Based on this analysis, provide a specific trading recommendation in the following format:
        
        TRADING SIGNAL: [LONG/SHORT/WAIT]
        ENTRY PRICE: [Specific price level]
        TAKE PROFIT: [Specific price level]
        STOP LOSS: [Specific price level]
        CONFIDENCE: [High/Medium/Low]
        REASONING: [Brief explanation of your analysis including key factors that led to this decision]
        
        Important: Be specific with exact price levels. Consider current market volatility and use proper risk management ratios.
        """
        
        return prompt
    
    def format_recent_candles(self, recent_data):
        """Format recent candle data for the prompt"""
        if recent_data is None or len(recent_data) == 0:
            return "No recent data available"
        
        formatted = []
        for i, (timestamp, candle) in enumerate(recent_data.iterrows()):
            candle_type = "🟢 BULLISH" if candle['Close'] > candle['Open'] else "🔴 BEARISH"
            formatted.append(f"Candle {i+1}: O:{candle['Open']:.5f} H:{candle['High']:.5f} L:{candle['Low']:.5f} C:{candle['Close']:.5f} {candle_type}")
        
        return "\n".join(formatted)
    
    def get_trading_signal(self, symbol, data, interval="5m"):
        """Get trading signal from Llama LLM"""
        if self.llm is None:
            return "LLM not initialized. Please check your Llama API key."
        
        market_structure = self.analyze_market_structure(data)
        prompt = self.create_market_analysis_prompt(symbol, data, market_structure, interval)
        
        if prompt is None:
            return "Insufficient data for analysis."
        
        # Print raw data to console for debugging
        print("\n" + "="*80)
        print("🔍 RAW DATA SENT TO LLM:")
        print("="*80)
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Data shape: {data.shape if data is not None else 'None'}")
        print(f"Market Structure: {market_structure}")
        print("\n📋 FULL PROMPT SENT TO LLM:")
        print("-"*80)
        print(prompt)
        print("-"*80)
        print("🤖 Sending to Llama 4...")
        print("="*80 + "\n")
        
        try:
            # Create a proper message format for ChatSambaNovaCloud
            messages = [("human", prompt)]
            response = self.llm.invoke(messages)
            
            # Print LLM response to console
            print("\n" + "="*80)
            print("🤖 LLM RESPONSE RECEIVED:")
            print("="*80)
            if hasattr(response, 'content'):
                print(response.content)
                result = response.content
            else:
                print(str(response))
                result = str(response)
            print("="*80 + "\n")
            
            return result
        except Exception as e:
            error_msg = f"Error getting LLM response: {e}"
            print(f"\n❌ ERROR: {error_msg}\n")
            return error_msg
    
    def create_chart(self, data, symbol, interval="5m"):
        """Create an interactive chart with technical indicators"""
        if data is None or len(data) == 0:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f'{symbol} - {interval.upper()} Chart', 'RSI', 'MACD'],
            row_heights=[0.7, 0.15, 0.15]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Add Bollinger Bands if available
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_Signal'], name='MACD Signal', line=dict(color='red')),
                row=3, col=1
            )
            if 'MACD_Histogram' in data.columns:
                colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
                fig.add_trace(
                    go.Bar(x=data.index, y=data['MACD_Histogram'], name='MACD Histogram', marker_color=colors),
                    row=3, col=1
                )
        
        fig.update_layout(
            title=f'{symbol} Trading Analysis',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
    
    def get_tradingview_symbol(self, yahoo_symbol):
        """Convert Yahoo Finance symbol to TradingView symbol"""
        symbol_mapping = {
            # Indices mapping
            "^NDX": "NASDAQ:NDX",        # NASDAQ 100
            "^IXIC": "NASDAQ:IXIC",      # NASDAQ Composite
            "^GSPC": "TVC:SPX",          # S&P 500
            "^DJI": "TVC:DJI",           # Dow Jones
            "^RUT": "TVC:RUT",           # Russell 2000
            "^VIX": "TVC:VIX",           # VIX
            "^FTSE": "TVC:UKX",          # FTSE 100
            "^GDAXI": "TVC:DAX",         # DAX
            "^N225": "TVC:NI225",        # Nikkei 225
            "^HSI": "TVC:HSI",           # Hang Seng
            "^FCHI": "TVC:CAC",          # CAC 40
            "^AXJO": "TVC:XJO",          # ASX 200
            
            # Commodities mapping
            "GC=F": "TVC:GOLD",          # Gold
            "SI=F": "TVC:SILVER",        # Silver  
            "CL=F": "TVC:USOIL",         # Crude Oil
            "NG=F": "TVC:NATURALGAS",    # Natural Gas
            "HG=F": "TVC:COPPER",        # Copper
            "PL=F": "TVC:PLATINUM",      # Platinum
            "PA=F": "TVC:PALLADIUM",     # Palladium
            
            # Forex mapping (remove =X)
            "EURUSD=X": "FX:EURUSD",
            "GBPUSD=X": "FX:GBPUSD", 
            "USDJPY=X": "FX:USDJPY",
            "USDCHF=X": "FX:USDCHF",
            "AUDUSD=X": "FX:AUDUSD",
            "USDCAD=X": "FX:USDCAD",
            "NZDUSD=X": "FX:NZDUSD",
            
            # Crypto mapping (replace -USD with USD)
            "BTC-USD": "BINANCE:BTCUSDT",
            "ETH-USD": "BINANCE:ETHUSDT",
            "ADA-USD": "BINANCE:ADAUSDT",
            "DOT-USD": "BINANCE:DOTUSDT", 
            "LINK-USD": "BINANCE:LINKUSDT",
            "UNI-USD": "BINANCE:UNIUSDT",
            "SOL-USD": "BINANCE:SOLUSDT",
            "AVAX-USD": "BINANCE:AVAXUSDT",
            "MATIC-USD": "BINANCE:MATICUSDT",
        }
        
        # Return mapped symbol or add NASDAQ prefix for stocks
        if yahoo_symbol in symbol_mapping:
            return symbol_mapping[yahoo_symbol]
        else:
            # For stocks, add NASDAQ prefix
            return f"NASDAQ:{yahoo_symbol}"

def main():
    st.set_page_config(
        page_title="AI Trading Bot with Llama 4",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🤖 AI Trading Bot with Llama 4")
    st.markdown("### Real-time Trading Signals powered by Llama 4")
    
    # Initialize trading bot
    if 'trading_bot' not in st.session_state:
        st.session_state.trading_bot = TradingBot()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Llama API Key",
            type="password",
            value=os.getenv("SAMBANOVA_API_KEY", ""),
            help="Enter your Llama API key from SambaNova Cloud"
        )
        
        if api_key:
            os.environ["SAMBANOVA_API_KEY"] = api_key
            st.session_state.trading_bot.sambanova_api_key = api_key
            st.session_state.trading_bot.setup_llm()
        
        st.markdown("---")
        
        # Trading pair selection
        st.subheader("📊 Trading Pair")
        
        pair_categories = {
            "Forex Major": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"],
            "Indices": {
                "NASDAQ": "^NDX",
                "S&P 500": "^GSPC",
                "Dow Jones": "^DJI",
                "Russell 2000": "^RUT",
                "VIX": "^VIX",
                "FTSE 100": "^FTSE",
                "DAX": "^GDAXI",
                "Nikkei 225": "^N225",
                "Hang Seng": "^HSI",
                "CAC 40": "^FCHI",
                "ASX 200": "^AXJO"
            },
            "Commodities": {
                "Gold": "GC=F",
                "Silver": "SI=F", 
                "Crude Oil": "CL=F",
                "Natural Gas": "NG=F",
                "Copper": "HG=F",
                "Platinum": "PL=F",
                "Palladium": "PA=F",
                "Corn": "ZC=F",
                "Wheat": "ZW=F",
                "Soybeans": "ZS=F"
            },
            "Crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "UNI-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"],
            "Stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "INTC"]
        }
        
        selected_category = st.selectbox("Select Category", list(pair_categories.keys()))
        
        # Handle different formats for different categories
        if selected_category == "Indices":
            # For indices, show the descriptive names but get the symbol
            indices_options = list(pair_categories[selected_category].keys())
            selected_index = st.selectbox("Select Trading Pair", indices_options)
            selected_pair = pair_categories[selected_category][selected_index]
        elif selected_category == "Commodities":
            # For commodities, show the descriptive names but get the symbol
            commodity_options = list(pair_categories[selected_category].keys())
            selected_commodity = st.selectbox("Select Trading Pair", commodity_options)
            selected_pair = pair_categories[selected_category][selected_commodity]
        else:
            # For other categories, use the list format as before
            selected_pair = st.selectbox("Select Trading Pair", pair_categories[selected_category])
        
        st.markdown("---")
        
        # Chart settings
        st.subheader("📈 Chart Settings")
        chart_period = st.selectbox("Period", ["1d", "5d", "1mo"], index=1)
        chart_interval = st.selectbox(
            "Chart Interval", 
            ["1m", "2m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=2,
            help="Pilih interval waktu untuk chart"
        )
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        if auto_refresh:
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 {selected_pair} Chart & Analysis")
        
        # Fetch and display data
        with st.spinner("Fetching market data..."):
            data = st.session_state.trading_bot.get_market_data(selected_pair, period=chart_period, interval=chart_interval)
        
        if data is not None and len(data) > 0:
            # Calculate technical indicators
            data_with_indicators = st.session_state.trading_bot.calculate_technical_indicators(data)
            
            # Create and display chart
            chart = st.session_state.trading_bot.create_chart(data_with_indicators, selected_pair, chart_interval)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Market data summary
            with st.expander("📊 Market Data Summary", expanded=False):
                latest = data_with_indicators.iloc[-1]
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Current Price", f"{latest['Close']:.5f}")
                
                with col_b:
                    price_change = latest['Close'] - data_with_indicators.iloc[-2]['Close']
                    st.metric("Price Change", f"{price_change:.5f}", f"{(price_change/data_with_indicators.iloc[-2]['Close']*100):.2f}%")
                
                with col_c:
                    if 'RSI' in data_with_indicators.columns and not pd.isna(latest['RSI']):
                        st.metric("RSI", f"{latest['RSI']:.2f}")
                
                with col_d:
                    st.metric("Volume", f"{latest['Volume']:,.0f}")
        else:
            st.error("Unable to fetch market data. Please try a different symbol.")
    
    with col2:
        st.subheader("🤖 AI Trading Signal")
        
        if st.button("🔍 Get Trading Signal", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your SambaNova API key in the sidebar.")
            elif data is not None:
                with st.spinner("🧠 Analyzing market with Llama AI..."):
                    signal = st.session_state.trading_bot.get_trading_signal(selected_pair, data_with_indicators, chart_interval)
                
                # Parse and display structured analysis
                if signal and "TRADING SIGNAL:" in signal:
                    # Parse the signal response
                    parsed_signal = parse_trading_signal(signal)
                    
                    # Display signal with enhanced UI
                    display_enhanced_signal(parsed_signal, selected_pair)
                    
                    # Show full analysis in expandable section
                    with st.expander("📋 Full AI Analysis", expanded=False):
                        st.text_area("Complete Analysis", signal, height=300)
                else:
                    # Fallback for unstructured response
                    st.markdown("### 📋 AI Analysis")
                    st.text_area("Analysis Result", signal, height=400)
            else:
                st.error("No data available for analysis.")
        
        # Risk Management Info
        with st.expander("⚠️ Risk Management", expanded=True):
            st.markdown("""
            **Important Reminders:**
            - Never risk more than 1-2% of your account per trade
            - Always use stop losses
            - This is AI-generated advice, not financial guidance
            - Do your own research and analysis
            - Consider market conditions and news events
            """)
        
        # TradingView Widget
        st.subheader("📈 TradingView Chart")
        tradingview_symbol = st.session_state.trading_bot.get_tradingview_symbol(selected_pair)
        
        tradingview_widget = f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "width": "100%",
            "height": "400",
            "symbol": "{tradingview_symbol}",
            "interval": "5",
            "timezone": "Etc/UTC",
            "theme": "light",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_widget"
          }}
          );
          </script>
        </div>
        """
        
        st.components.v1.html(tradingview_widget, height=420)

    # Add explanation section at the bottom
    st.markdown("---")
    st.header("🧠 Cara Kerja AI Trading Bot")
    
    with st.expander("📖 Logic & Alur Kerja Bot (Klik untuk melihat detail)", expanded=False):
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("""
            ### 🔄 **Alur Kerja Bot (Step by Step):**
            
            **1. 📊 Data Collection**
            - Bot mengambil data market real-time dari Yahoo Finance
            - Mendapatkan data OHLCV (Open, High, Low, Close, Volume)
            - Data diambil sesuai interval yang dipilih (1m-1d)
            
            **2. 📈 Technical Analysis**
            - **RSI (Relative Strength Index)**: Mengukur momentum harga
              - RSI > 70 = Overbought (kemungkinan turun)
              - RSI < 30 = Oversold (kemungkinan naik)
            
            - **MACD (Moving Average Convergence Divergence)**: 
              - MACD line vs Signal line untuk trend direction
              - Histogram untuk strength momentum
            
            - **Bollinger Bands**: Mengukur volatilitas
              - Harga di upper band = resistance
              - Harga di lower band = support
            
            - **Moving Averages**: SMA 20 & SMA 50
              - Golden cross (SMA20 > SMA50) = bullish
              - Death cross (SMA20 < SMA50) = bearish
            
            **3. 🏗️ Market Structure Analysis**
            - Identifikasi support & resistance levels
            - Analisis recent highs & lows
            - Pattern recognition untuk price action
            """)
        
        with col_right:
            st.markdown("""
            **4. 🤖 AI Analysis dengan Llama 4**
            - Bot mengirim semua data ke AI Llama 4
            - AI menganalisis kombinasi semua indikator
            - AI mempertimbangkan:
              - Current price action
              - Technical indicator signals
              - Market structure levels
              - Risk-reward ratio
            
            **5. 📋 Signal Generation**
            - AI menghasilkan structured recommendation:
              - **TRADING SIGNAL**: LONG/SHORT/WAIT
              - **ENTRY PRICE**: Level masuk yang optimal
              - **TAKE PROFIT**: Target profit
              - **STOP LOSS**: Level cut loss
              - **CONFIDENCE**: High/Medium/Low
              - **REASONING**: Penjelasan analisis
            
            **6. 🎯 Risk Management**
            - AI mempertimbangkan proper risk-reward ratio
            - Stop loss biasanya 1-2% dari entry
            - Take profit biasanya 2-3x stop loss distance
            - Confidence level berdasarkan strength signal
            
            **7. 📊 Real-time Updates**
            - Data diupdate setiap kali user refresh
            - Chart menampilkan analisis visual
            - TradingView widget untuk konfirmasi
            """)
    
    with st.expander("⚙️ Technical Indicators Explained", expanded=False):
        indicator_col1, indicator_col2 = st.columns(2)
        
        with indicator_col1:
            st.markdown("""
            ### 📊 **RSI (Relative Strength Index)**
            ```
            Formula: RSI = 100 - (100 / (1 + RS))
            RS = Average Gain / Average Loss (14 periods)
            ```
            - **Fungsi**: Mengukur kecepatan perubahan harga
            - **Range**: 0-100
            - **Overbought**: > 70 (sell signal)
            - **Oversold**: < 30 (buy signal)
            - **Divergence**: RSI vs Price = reversal signal
            
            ### 📈 **MACD (Moving Average Convergence Divergence)**
            ```
            MACD Line = EMA(12) - EMA(26)
            Signal Line = EMA(9) of MACD Line
            Histogram = MACD Line - Signal Line
            ```
            - **Bullish**: MACD > Signal Line
            - **Bearish**: MACD < Signal Line
            - **Momentum**: Histogram strength
            """)
        
        with indicator_col2:
            st.markdown("""
            ### 📏 **Bollinger Bands**
            ```
            Middle Band = SMA(20)
            Upper Band = SMA(20) + (2 × Standard Deviation)
            Lower Band = SMA(20) - (2 × Standard Deviation)
            ```
            - **Volatility**: Band width = volatility level
            - **Support**: Lower band acts as support
            - **Resistance**: Upper band acts as resistance
            - **Squeeze**: Narrow bands = breakout coming
            
            ### 📊 **Moving Averages**
            ```
            SMA = (Sum of prices) / Number of periods
            ```
            - **SMA 20**: Short-term trend
            - **SMA 50**: Medium-term trend
            - **Golden Cross**: SMA20 crosses above SMA50
            - **Death Cross**: SMA20 crosses below SMA50
            """)
    
    with st.expander("🧠 AI Decision Making Process", expanded=False):
        st.markdown("""
        ### 🤖 **Bagaimana AI Llama 4 Menganalisis Data:**
        
        **1. Data Input Processing:**
        ```
        Input yang dikirim ke AI:
        ├── Current Market Conditions (Price, Volume, Change%)
        ├── Technical Indicators (RSI, MACD, Bollinger Bands)
        ├── Market Structure (Support/Resistance levels)
        ├── Recent Price Action (Last 5 candles)
        └── Timeframe Context (1m, 5m, 1h, etc.)
        ```
        
        **2. AI Analysis Framework:**
        - **Trend Analysis**: Apakah market sedang uptrend, downtrend, atau sideways?
        - **Momentum Check**: Seberapa kuat momentum saat ini?
        - **Support/Resistance**: Apakah harga mendekati level penting?
        - **Risk Assessment**: Berapa risk-reward ratio yang optimal?
        - **Confluence**: Berapa banyak indikator yang confirm signal?
        
        **3. Signal Confidence Levels:**
        - **HIGH**: 3+ indikator konfirmasi + clear trend
        - **MEDIUM**: 2 indikator konfirmasi + moderate trend
        - **LOW**: 1 indikator konfirmasi atau mixed signals
        
        **4. Entry/Exit Logic:**
        ```
        Entry Price = Current price ± spread consideration
        Stop Loss = Recent support/resistance ± buffer
        Take Profit = Risk-reward ratio 1:2 atau 1:3
        ```
        
        **5. Risk Management Rules:**
        - Maximum risk per trade: 1-2% of account
        - Always use stop loss
        - Position sizing based on volatility
        - No trading during major news events (if detected)
        """)
    
    # Add disclaimer
    # st.markdown("---")
    # st.warning("""
    # ⚠️ **DISCLAIMER**: 
    # - Bot ini adalah tools edukasi dan analisis, bukan financial advice
    # - Selalu lakukan riset sendiri sebelum trading
    # - Trading memiliki risiko kehilangan modal
    # - Gunakan risk management yang proper
    # - Past performance tidak menjamin future results
    # """)

if __name__ == "__main__":
    main()