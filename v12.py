import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import argrelextrema

# --- PAGE CONFIG ---
st.set_page_config(page_title="US Stocks Analyzer", layout="wide", page_icon="📈")

# --- THEME ENGINE ---
if 'is_dark' not in st.session_state:
    st.session_state['is_dark'] = True

def toggle_theme():
    st.session_state['is_dark'] = not st.session_state['is_dark']

# Sidebar Toggle
st.sidebar.header("⚙️ Settings")
is_dark = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state['is_dark'], on_change=toggle_theme)

# --- CSS & COLORS ---
if is_dark:
    # DARK PALETTE
    bg_color = "#0E1117"
    card_color = "#262730"
    text_color = "#FAFAFA"
    chart_bg = "#0E1117"
    grid_color = "#444444" 
    plotly_template = "plotly_dark"
    tv_theme = "dark"
    
    # Aggressive CSS for Dark Mode (Fixed Buttons & Inputs)
    st.markdown(f"""
        <style>
            .stApp {{ background-color: {bg_color} !important; color: {text_color} !important; }}
            [data-testid="stSidebar"] {{ background-color: {bg_color} !important; }}
            [data-testid="stSidebar"] * {{ color: {text_color} !important; }}
            [data-testid="stHeader"] {{ background-color: {bg_color} !important; }}
            
            /* Tables */
            .stDataFrame {{ background-color: {card_color} !important; }}
            [data-testid="stDataFrameResizable"] {{ background-color: {card_color} !important; }}
            
            /* FIXED BUTTONS */
            .stButton > button {{
                color: white !important; 
                background-color: {card_color} !important; 
                border: 1px solid #4B4B4B !important;
            }}
            .stButton > button:hover {{
                border-color: #00FF00 !important;
                color: #00FF00 !important;
            }}
            
            /* FIXED SELECTBOX */
            div[data-baseweb="select"] > div {{
                background-color: {card_color} !important;
                color: {text_color} !important;
                border-color: #4B4B4B !important;
            }}
            div[data-baseweb="select"] span {{
                color: {text_color} !important;
            }}
            /* Dropdown options */
            ul[data-testid="stSelectboxVirtualDropdown"] {{
                background-color: {card_color} !important;
            }}
            li[role="option"] {{
                color: {text_color} !important;
                background-color: {card_color} !important;
            }}
            li[role="option"]:hover {{
                background-color: #444444 !important;
            }}
            
            h1, h2, h3, h4, h5, h6, p, label, span, div {{ color: {text_color} !important; }}
            .stMetricLabel {{ color: #A0A0A0 !important; }}
        </style>
    """, unsafe_allow_html=True)
else:
    # LIGHT PALETTE
    bg_color = "#FFFFFF"
    card_color = "#F0F2F6"
    text_color = "#31333F"
    chart_bg = "#FFFFFF"
    grid_color = "#E6E6EA" 
    plotly_template = "plotly_white"
    tv_theme = "light"
    
    st.markdown(f"""
        <style>
            .stApp {{ background-color: {bg_color} !important; color: {text_color} !important; }}
            [data-testid="stSidebar"] {{ background-color: #F8F9FB !important; }}
            [data-testid="stHeader"] {{ background-color: {bg_color} !important; }}
            .stDataFrame {{ background-color: {bg_color} !important; }}
            
            /* Light Mode Buttons */
            div.stButton > button {{ 
                color: {text_color} !important; 
                border-color: #D3D3D3 !important; 
                background-color: #FFFFFF !important; 
            }}
            h1, h2, h3, h4, h5, h6, p, label {{ color: {text_color} !important; }}
        </style>
    """, unsafe_allow_html=True)

# --- COLOR CUSTOMIZATION ---
with st.sidebar.expander("🎨 Chart Colors", expanded=False):
    c_ema200 = st.color_picker("EMA 200 (Trend)", "#FFD700")  # Gold
    c_ema50 = st.color_picker("EMA 50 (Fast)", "#FFFFFF" if is_dark else "#000000")
    c_ut_buy = st.color_picker("UT Bot Buy Line", "#00FF00")  # Green
    c_ut_sell = st.color_picker("UT Bot Sell Line", "#FF0000") # Red
    c_sig_buy = st.color_picker("Buy Signal Marker", "#00FF00")
    c_sig_sell = st.color_picker("Sell Signal Marker", "#FF0000")

# --- CHART OVERLAYS ---
st.sidebar.subheader("🖼️ Overlays")
show_patterns = st.sidebar.toggle("🧩 Show Patterns", value=True)

# --- 0. CONFIGURATION ---
TICKERS = [
    'AAPL', 'TSLA', 'NVDA', 'AVGO', 'AZN', 'CSCO', 'WBA', 'ASML', 'ADBE', 'GILD', 
    'PANW', 'TXN', 'QCOM', 'SBUX', 'REGN', 'ISRG', 'ADI', 'AMAT', 'MU', 'TEAM', 
    'ORLY', 'LRCX', 'SNPS', 'ADSK', 'CTAS', 'FTNT', 'CSX', 'BIIB', 'DXCM', 'MCHP', 
    'KLAC', 'LULU', 'NXPI', 'CRWD', 'ILMN', 'MRVL', 'ODFL', 'ROST', 'IDXX', 'CPRT', 
    'FAST', 'VRSN', 'ANSS', 'ALGN', 'SWKS', 'DOCU', 'OKTA', 'GOOGL', 'AMZN', 'META', 
    'PEP', 'AMD', 'DLTR', 'JD', 'MDLZ', 'MAR', 'CDNS', 'COST', 'NFLX', 'INTU', 
    'BKNG', 'NTES', 'PAYX', 'EA', 'EBAY', 'MTCH', 'TMUS', 'CMCSA', 'HON', 'PYPL', 
    'ADP', 'FISV', 'MRNA', 'AEP', 'KHC', 'MELI', 'EXC', 'XEL', 'PCAR', 'CEG', 
    'VRSK', 'SIRI', 'LCID', 'AMGN', 'INTC', 'CHTR', 'VRTX', 'ABNB', 'KDP', 'MNST', 
    'WDAY', 'BIDU', 'CTSH', 'DDOG', 'ZS', 'ZM', 'MSFT', 'PDD'
]

# --- 1. PATTERN RECOGNITION ENGINE (V3 - ROBUST) ---
def identify_patterns(df):
    """
    V3: Uses rolling local extrema to find true pivot points.
    Far more accurate than min/max over arbitrary windows.
    """
    if len(df) < 50: return None
    
    # We analyze the last 50 candles
    window = df.tail(50).copy()
    
    # 1. Find Pivot Points (Local Minima/Maxima with order 3)
    # This finds candles that are lower than 3 neighbors on each side
    lows = window['Low'].values
    highs = window['High'].values
    
    # Indices of local bottoms and tops within the window
    bottom_indices = argrelextrema(lows, np.less, order=3)[0]
    top_indices = argrelextrema(highs, np.greater, order=3)[0]
    
    pat_name = None
    sentiment = None
    price_level = 0
    idx_of_pat = None

    # --- A. DOUBLE BOTTOM (W) ---
    # We need at least 2 distinct bottoms
    if len(bottom_indices) >= 2:
        # Check the last two bottoms
        b1_idx = bottom_indices[-2]
        b2_idx = bottom_indices[-1]
        
        price_b1 = lows[b1_idx]
        price_b2 = lows[b2_idx]
        
        # Logic:
        # 1. Prices are close (within 1.5%)
        # 2. Time gap is decent (> 5 bars)
        # 3. The second bottom happened recently (in the last 15 bars)
        if abs(price_b1 - price_b2) / price_b1 < 0.015:
            if (b2_idx - b1_idx) > 5:
                # Is the pattern still relevant? (Price shouldn't have skyrocketed 5% away already)
                curr_price = window['Close'].iloc[-1]
                if curr_price < price_b2 * 1.05: 
                    pat_name = "Double Bottom (W)"
                    sentiment = "Pos"
                    idx_of_pat = window.index[b2_idx]
                    price_level = price_b2

    # --- B. DOUBLE TOP (M) ---
    if not pat_name and len(top_indices) >= 2:
        t1_idx = top_indices[-2]
        t2_idx = top_indices[-1]
        
        price_t1 = highs[t1_idx]
        price_t2 = highs[t2_idx]
        
        if abs(price_t1 - price_t2) / price_t1 < 0.015:
            if (t2_idx - t1_idx) > 5:
                curr_price = window['Close'].iloc[-1]
                if curr_price > price_t2 * 0.95:
                    pat_name = "Double Top (M)"
                    sentiment = "Neg"
                    idx_of_pat = window.index[t2_idx]
                    price_level = price_t2

    # --- C. CANDLESTICK PATTERNS (Trend Context Aware) ---
    # Only checks the VERY LAST completed candle
    if not pat_name:
        last = window.iloc[-1]
        prev = window.iloc[-2]
        
        body = abs(last['Close'] - last['Open'])
        upper_wick = last['High'] - max(last['Close'], last['Open'])
        lower_wick = min(last['Close'], last['Open']) - last['Low']
        
        # Calculate Trend (SMA 20 Slope)
        sma20 = window['Close'].rolling(20).mean()
        is_downtrend = sma20.iloc[-1] < sma20.iloc[-5]
        is_uptrend = sma20.iloc[-1] > sma20.iloc[-5]
        
        # Hammer (Must be in downtrend)
        if is_downtrend and (lower_wick > 2 * body) and (upper_wick < body * 0.5):
            pat_name = "Hammer 🔨"; sentiment = "Pos"; idx_of_pat = last.name; price_level = last['Low']

        # Shooting Star (Must be in uptrend)
        elif is_uptrend and (upper_wick > 2 * body) and (lower_wick < body * 0.5):
            pat_name = "Shooting Star 🌠"; sentiment = "Neg"; idx_of_pat = last.name; price_level = last['High']
            
        # Bullish Engulfing
        elif is_downtrend and (prev['Close'] < prev['Open']) and (last['Close'] > last['Open']):
            if (last['Open'] < prev['Close']) and (last['Close'] > prev['Open']):
                pat_name = "Bullish Engulfing 🐂"; sentiment = "Pos"; idx_of_pat = last.name; price_level = last['Low']

    if pat_name:
        return {'name': pat_name, 'sentiment': sentiment, 'index': idx_of_pat, 'price': price_level}
    return None

# --- 2. STRATEGY & CALCULATIONS ---
def calculate_expert_strategy(df, sensitivity=2, atr_period=1):
    df['ATR_UT'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)
    df['src'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['nLoss'] = sensitivity * df['ATR_UT']
    
    src = df['src'].values; nLoss = df['nLoss'].values
    trailing_stop = np.zeros(len(df))
    
    for i in range(atr_period, len(df)):
        prev_stop = trailing_stop[i-1]
        curr_src = src[i]; prev_src = src[i-1]
        loss = nLoss[i]
        
        if (curr_src > prev_stop) and (prev_src > prev_stop):
            trailing_stop[i] = max(prev_stop, curr_src - loss)
        elif (curr_src < prev_stop) and (prev_src < prev_stop):
            trailing_stop[i] = min(prev_stop, curr_src + loss)
        elif (curr_src > prev_stop):
            trailing_stop[i] = curr_src - loss
        else: 
            trailing_stop[i] = curr_src + loss
            
    df['Trailing_Stop'] = trailing_stop
    df['UT_Trend'] = np.where(df['src'] > df['Trailing_Stop'], 1, -1)

    stoch = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
    if stoch is not None:
        df['Stoch_K'] = stoch['STOCHRSIk_14_14_3_3']
        df['Stoch_D'] = stoch['STOCHRSId_14_14_3_3']
    else:
        df['Stoch_K'] = 50; df['Stoch_D'] = 50

    df['EMA_200'] = ta.ema(df['Close'], length=200)

    df['Buy_Signal'] = (
        (df['UT_Trend'] == 1) & (df['Close'] > df['EMA_200']) &
        (df['Stoch_K'].shift(1) < 20) & (df['Stoch_K'] > df['Stoch_D']) & 
        (df['Stoch_K'] > df['Stoch_K'].shift(1))
    )
    df['Sell_Signal'] = (
        (df['UT_Trend'] == -1) & (df['Close'] < df['EMA_200']) &
        (df['Stoch_K'].shift(1) > 80) & (df['Stoch_K'] < df['Stoch_D']) &
        (df['Stoch_K'] < df['Stoch_K'].shift(1))
    )
    return df

# --- 3. INDICATOR PROCESSING ---
def process_stock_data(df):
    df = calculate_expert_strategy(df)
    
    # Indicators
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['ADL'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
    
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    if adx is not None: df = pd.concat([df, adx], axis=1)
    
    macd = ta.macd(df['Close'])
    if macd is not None: df = pd.concat([df, macd], axis=1)
    
    try:
        ichi = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
        df = pd.concat([df, ichi], axis=1)
    except Exception: pass

    # Detect Pattern
    pattern_data = identify_patterns(df)

    # --- INTERPRETATIONS ---
    last = df.iloc[-1]
    interp_list = {'Positive': [], 'Neutral': [], 'Negative': []}
    chart_text = {}

    # 1. UT Bot
    if last['UT_Trend'] == 1: 
        msg = "UT Bot: BUY Zone"; interp_list['Positive'].append(msg); chart_text['UT'] = f"🟢 {msg}"
    else: 
        msg = "UT Bot: SELL Zone"; interp_list['Negative'].append(msg); chart_text['UT'] = f"🔴 {msg}"

    # 2. EMA
    if last['Close'] > last['EMA_200']: 
        msg = "Price > EMA 200 (Bullish)"; interp_list['Positive'].append(msg); chart_text['EMA'] = f"🟢 {msg}"
    else: 
        msg = "Price < EMA 200 (Bearish)"; interp_list['Negative'].append(msg); chart_text['EMA'] = f"🔴 {msg}"

    # 3. RSI
    if last['RSI'] < 30: 
        msg = f"RSI {last['RSI']:.0f} (Oversold)"; interp_list['Positive'].append(msg); chart_text['RSI'] = f"🟢 {msg}"
    elif last['RSI'] > 70: 
        msg = f"RSI {last['RSI']:.0f} (Overbought)"; interp_list['Negative'].append(msg); chart_text['RSI'] = f"🔴 {msg}"
    else: 
        msg = f"RSI {last['RSI']:.0f} (Neutral)"; interp_list['Neutral'].append(msg); chart_text['RSI'] = f"🟡 {msg}"

    # 4. MACD
    if 'MACD_12_26_9' in last:
        if last['MACD_12_26_9'] > last['MACDs_12_26_9']: 
            msg = "MACD Bullish Cross"; interp_list['Positive'].append(msg); chart_text['MACD'] = f"🟢 {msg}"
        else: 
            msg = "MACD Bearish Cross"; interp_list['Negative'].append(msg); chart_text['MACD'] = f"🔴 {msg}"

    # 5. Stoch RSI
    if last['Stoch_K'] < 20: 
        msg = "Stoch RSI Oversold"; interp_list['Positive'].append(msg); chart_text['STOCH'] = f"🟢 {msg}"
    elif last['Stoch_K'] > 80: 
        msg = "Stoch RSI Overbought"; interp_list['Negative'].append(msg); chart_text['STOCH'] = f"🔴 {msg}"
    else: 
        msg = "Stoch RSI Neutral"; interp_list['Neutral'].append(msg); chart_text['STOCH'] = f"🟡 {msg}"

    # 6. Ichimoku
    if 'ISA_9' in last and 'ISB_26' in last:
        cloud_top = max(last['ISA_9'], last['ISB_26'])
        if last['Close'] > cloud_top: 
            msg = "Price Above Cloud"; interp_list['Positive'].append(msg); chart_text['ICHI'] = f"🟢 {msg}"
        elif last['Close'] < min(last['ISA_9'], last['ISB_26']): 
            msg = "Price Below Cloud"; interp_list['Negative'].append(msg); chart_text['ICHI'] = f"🔴 {msg}"
        else: 
            msg = "Price Inside Cloud"; interp_list['Neutral'].append(msg); chart_text['ICHI'] = f"🟡 {msg}"
    else: chart_text['ICHI'] = "⚪ N/A"

    # 7. VWAP
    if 'VWAP' in last:
        if last['Close'] > last['VWAP']: 
            msg = "Price > VWAP (Bullish)"; interp_list['Positive'].append(msg); chart_text['VWAP'] = f"🟢 {msg}"
        else: 
            msg = "Price < VWAP (Bearish)"; interp_list['Negative'].append(msg); chart_text['VWAP'] = f"🔴 {msg}"

    # 8. ADX
    if 'ADX_14' in last:
        if last['ADX_14'] > 25: 
            msg = f"Strong Trend (ADX {last['ADX_14']:.0f})"; interp_list['Positive'].append(msg); chart_text['ADX'] = f"🟢 {msg}"
        else: 
            msg = f"Weak Trend (ADX {last['ADX_14']:.0f})"; interp_list['Neutral'].append(msg); chart_text['ADX'] = f"🟡 {msg}"

    # 9. CMF
    if 'CMF' in last:
        if last['CMF'] > 0: 
            msg = "Inflow (Buying)"; interp_list['Positive'].append(msg); chart_text['CMF'] = f"🟢 {msg}"
        else: 
            msg = "Outflow (Selling)"; interp_list['Negative'].append(msg); chart_text['CMF'] = f"🔴 {msg}"

    # 10. CCI
    if 'CCI' in last:
        if last['CCI'] < -100: 
            msg = "CCI Oversold"; interp_list['Positive'].append(msg); chart_text['CCI'] = f"🟢 {msg}"
        elif last['CCI'] > 100: 
            msg = "CCI Overbought"; interp_list['Negative'].append(msg); chart_text['CCI'] = f"🔴 {msg}"
        else: 
            msg = "CCI Neutral"; interp_list['Neutral'].append(msg); chart_text['CCI'] = f"🟡 {msg}"

    # 11. OBV
    if 'OBV' in last and 'OBV' in df.iloc[-2]:
        if last['OBV'] > df.iloc[-2]['OBV']: 
            msg = "OBV Rising"; interp_list['Positive'].append(msg); chart_text['OBV'] = f"🟢 {msg}"
        else: 
            msg = "OBV Falling"; interp_list['Negative'].append(msg); chart_text['OBV'] = f"🔴 {msg}"
            
    # 12. ADL
    if 'ADL' in last and 'ADL' in df.iloc[-2]:
        if last['ADL'] > df.iloc[-2]['ADL']: 
            msg = "ADL Rising"; interp_list['Positive'].append(msg); chart_text['ADL'] = f"🟢 {msg}"
        else: 
            msg = "ADL Falling"; interp_list['Negative'].append(msg); chart_text['ADL'] = f"🔴 {msg}"

    # Win Rate
    pos_count = len(interp_list['Positive'])
    wins, losses, trades = 0, 0, 0
    entries = df.index[df['Buy_Signal']]
    shorts = df.index[df['Sell_Signal']]
    target, stop = 0.010, 0.020
    
    for t in entries:
        if t == df.index[-1]: continue
        p = df.loc[t]['Close']
        future = df.loc[t:].iloc[1:]
        for _, r in future.iterrows():
            if r['High'] >= p*(1+target): wins+=1; trades+=1; break
            if r['Low'] <= p*(1-stop): losses+=1; trades+=1; break
            
    for t in shorts:
        if t == df.index[-1]: continue
        p = df.loc[t]['Close']
        future = df.loc[t:].iloc[1:]
        for _, r in future.iterrows():
            if r['Low'] <= p*(1-target): wins+=1; trades+=1; break
            if r['High'] >= p*(1+stop): losses+=1; trades+=1; break
            
    wr = (wins/trades*100) if trades > 0 else 0
    return df, wr, trades, pos_count, interp_list, chart_text, pattern_data

# --- 4. DECISION ---
def get_decision(df):
    last = df.iloc[-1]
    if last['Buy_Signal']: return "💎 BUY DIP", "green"
    elif last['Sell_Signal']: return "🩸 SELL RALLY", "red"
    elif last['UT_Trend'] == 1: return "✅ UPTREND", "lightgreen"
    return "⚠️ DOWNTREND", "gray"

# --- 5. DATA PIPELINE ---
@st.cache_data(ttl=3600, show_spinner="Analyzing Market...")
def get_data(tickers):
    bulk = yf.download(tickers, period="1y", interval="1h", group_by='ticker', progress=False, threads=True)
    processed = {}
    summ = []
    failed = []

    for t in tickers:
        try:
            df = bulk[t].copy()
            if df.empty or len(df) < 100: continue
            df.dropna(inplace=True)

            # Drop the current, still-forming candle (its High/Low/Close are
            # mid-bar and will keep changing until the hour closes), otherwise
            # signals/patterns are computed on incomplete data and repaint.
            if not df.empty:
                last_ts = df.index[-1]
                now = pd.Timestamp.now(tz=last_ts.tzinfo)
                if last_ts + pd.Timedelta(hours=1) > now:
                    df = df.iloc[:-1]
            if df.empty or len(df) < 100: continue

            df, wr, tr, score, interp_list, chart_text, pattern = process_stock_data(df)
            processed[t] = {'df': df, 'wr': wr, 'tr': tr, 'interp_list': interp_list, 'chart_text': chart_text, 'pattern': pattern}
            
            last = df.iloc[-1]
            dec, col = get_decision(df)
            
            live = "💎 LONG" if last['Buy_Signal'] else ("🩸 SHORT" if last['Sell_Signal'] else "-")
            
            pat_str = pattern['name'] if pattern else "None"
            
            summ.append({
                "TICKER": t,
                "PRICE": f"{last['Close']:.2f}",
                "PATTERN": pat_str,
                "LIVE": live,
                "WIN RATE": f"{int(wr)}% ({tr})",
                "DECISION": dec,
                "SCORE": f"{score}/12", # Updated Score
                "RSI": f"{int(last['RSI'])}"
            })
        except Exception:
            failed.append(t)

    return processed, pd.DataFrame(summ), failed

# --- MAIN APP ---
st.title("📈 US Stocks Analyzer")

if 'chart_range' not in st.session_state: st.session_state['chart_range'] = '1M'

data, df_summ, failed_tickers = get_data(TICKERS)

# --- SCANNER ---
st.subheader("📡 Market Scanner")
# Average over stocks with at least one closed backtest trade. (Previously this
# filtered on the substring "0%" in the display string, which also excluded
# every stock whose win rate ended in 0 - e.g. 10%, 20%, ... 100% - skewing
# the average.)
valid_wr = [d['wr'] for d in data.values() if d['tr'] > 0]
avg_wr = sum(valid_wr)/len(valid_wr) if valid_wr else 0
st.info(f"📊 **System Avg Win Rate:** {int(avg_wr)}%")
if failed_tickers:
    st.warning(f"⚠️ {len(failed_tickers)} ticker(s) failed to load and are excluded from the scanner: {', '.join(failed_tickers)}")

def style_table(df):
    def highlight_live(val):
        if 'LONG' in str(val) or 'BUY' in str(val): return 'background-color: #00FF00; color: black; font-weight: bold'
        if 'SHORT' in str(val) or 'SELL' in str(val): return 'background-color: #FF4444; color: white; font-weight: bold'
        return ''
    
    # Use Pandas Styler to Force Colors
    styler = df.style.map(highlight_live, subset=['LIVE', 'DECISION'])
    if is_dark:
        styler.set_properties(**{'background-color': '#262730', 'color': 'white', 'border-color': '#444444'})
    else:
        styler.set_properties(**{'background-color': '#FFFFFF', 'color': 'black', 'border-color': '#E6E6EA'})
    return styler

st.dataframe(style_table(df_summ), use_container_width=True)
st.divider()

# --- DEEP DIVE ---
selected = st.selectbox("Select Stock", sorted(data.keys()))

if selected:
    d = data[selected]
    df = d['df']
    interp_list = d['interp_list']
    ct = d['chart_text']
    pat = d['pattern']
    last = df.iloc[-1]
    
    if last['Buy_Signal']: st.success(f"🚨 LIVE BUY SIGNAL: {selected}")
    elif last['Sell_Signal']: st.error(f"🚨 LIVE SELL SIGNAL: {selected}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${last['Close']:.2f}")
    c2.metric("Win Rate", f"{int(d['wr'])}%", f"{d['tr']} Trades")
    c3.metric("Pos Indicators", f"{len(interp_list['Positive'])}")
    dec, _ = get_decision(df)
    c4.metric("Strategy", dec)

    # --- 1. TRADINGVIEW CHART ---
    st.markdown("### 🖼️ TradingView Chart")
    tv_theme_str = "dark" if is_dark else "light"
    components.html(f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{selected}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "{tv_theme_str}",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """, height=500)
    
    st.divider()

    # --- 2. PYTHON STRATEGY CHART ---
    st.markdown("### 📊 Strategy Chart (UT Bot & Patterns)")
    
    # CHART CONTROLS
    st.write("#### 🔎 Zoom")
    cols = st.columns(5)
    ranges = {'1W': '1W', '1M': '1M', '3M': '3M', '6M': '6M', 'ALL': 'ALL'}
    for i, (k,v) in enumerate(ranges.items()):
        if cols[i].button(k): st.session_state['chart_range'] = v
        
    df_c = df.copy()
    if st.session_state['chart_range'] == '1W': df_c = df_c.tail(35)
    elif st.session_state['chart_range'] == '1M': df_c = df_c.tail(150)
    elif st.session_state['chart_range'] == '3M': df_c = df_c.tail(450)
    elif st.session_state['chart_range'] == '6M': df_c = df_c.tail(900)
    elif st.session_state['chart_range'] == 'ALL': df_c = df_c # All Data
    
    idx = df_c.index.strftime('%Y-%m-%d %H:%M')
    
    # --- HELPER FOR CHART LAYOUT ---
    def update_fig(figure, title_txt=""):
        final_title = "" if title_txt == "Main" else title_txt
        figure.update_xaxes(type='category', nticks=10, showgrid=False)
        figure.update_layout(
            height=300 if title_txt != "Main" else 600,
            template=plotly_template,
            paper_bgcolor=bg_color, # Force Background
            plot_bgcolor=bg_color,  # Force Plot Area
            title_text=final_title,
            title_font_color=text_color,
            font=dict(color=text_color), # Legend/Axis Text Color Force
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="left", 
                x=0, 
                font=dict(color=text_color)
            )
        )
        return figure

    # MAIN CHART
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=idx, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name='Price'), row=1, col=1)
    
    # User-Colored Lines
    fig.add_trace(go.Scatter(x=idx, y=df_c['EMA_200'], line=dict(color=c_ema200, width=2), name='EMA 200'), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=df_c['EMA_50'], line=dict(color=c_ema50, width=1), name='EMA 50'), row=1, col=1)
    
    stop_g = df_c['Trailing_Stop'].where(df_c['UT_Trend']==1)
    stop_r = df_c['Trailing_Stop'].where(df_c['UT_Trend']==-1)
    fig.add_trace(go.Scatter(x=idx, y=stop_g, line=dict(color=c_ut_buy), name='UT Buy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=stop_r, line=dict(color=c_ut_sell), name='UT Sell'), row=1, col=1)
    
    # Signals
    buys = df_c[df_c['Buy_Signal']]
    sells = df_c[df_c['Sell_Signal']]
    if not buys.empty: fig.add_trace(go.Scatter(x=buys.index.strftime('%Y-%m-%d %H:%M'), y=buys['Low']*0.99, mode='markers', marker=dict(symbol='diamond', size=12, color=c_sig_buy), name='BUY'), row=1, col=1)
    if not sells.empty: fig.add_trace(go.Scatter(x=sells.index.strftime('%Y-%m-%d %H:%M'), y=sells['High']*1.01, mode='markers', marker=dict(symbol='diamond', size=12, color=c_sig_sell), name='SELL'), row=1, col=1)
    
    # Pattern Annotation (Main Chart)
    if show_patterns and pat:
        try:
            # Check if pattern index is in the current zoomed view
            pat_idx_str = pat['index'].strftime('%Y-%m-%d %H:%M')
            if pat_idx_str in idx.values:
                # Sentiment Color
                sent_color = "#00FF00" if pat['sentiment'] == "Pos" else "#FF0000"
                
                fig.add_annotation(
                    x=pat_idx_str, y=pat['price'],
                    text=f"<b>{pat['name']}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=sent_color,
                    bgcolor=bg_color,
                    bordercolor=sent_color,
                    font=dict(color=sent_color)
                )
        except: pass

    fig.add_trace(go.Scatter(x=idx, y=df_c['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', row=2, col=1); fig.add_hline(y=70, line_dash='dash', row=2, col=1)
    
    update_fig(fig, "Main")
    st.plotly_chart(fig, use_container_width=True, key=f"main_{plotly_template}_{selected}")
    
    # --- FULL INDICATOR GRID (Consolidated 6 Rows) ---
    st.markdown("### 📊 Advanced Technicals")
    
    # Row 1: Stoch RSI | Ichimoku
    r1a, r1b = st.columns(2)
    with r1a:
        f = go.Figure()
        f.add_trace(go.Scatter(x=idx, y=df_c['Stoch_K'], name='K', line=dict(color='cyan')))
        f.add_trace(go.Scatter(x=idx, y=df_c['Stoch_D'], name='D', line=dict(color='orange')))
        f.add_hline(y=20, line_color='green'); f.add_hline(y=80, line_color='red')
        st.plotly_chart(update_fig(f, "Stochastic RSI"), use_container_width=True, key=f"st_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('STOCH', 'N/A')}")
    with r1b:
        f = go.Figure()
        if 'ISA_9' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['Close'], name='Close'))
            f.add_trace(go.Scatter(x=idx, y=df_c['ISA_9'], name='A', line=dict(width=0), showlegend=False))
            f.add_trace(go.Scatter(x=idx, y=df_c['ISB_26'], name='B', fill='tonexty', fillcolor='rgba(0,0,250,0.1)', line=dict(width=0), showlegend=False))
        st.plotly_chart(update_fig(f, "Ichimoku Cloud"), use_container_width=True, key=f"ich_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('ICHI', 'N/A')}")

    # Row 2: VWAP | SMA/EMA Trends
    r2a, r2b = st.columns(2)
    with r2a:
        f = go.Figure()
        if 'VWAP' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['Close'], name='Price'))
            f.add_trace(go.Scatter(x=idx, y=df_c['VWAP'], name='VWAP', line=dict(color='orange')))
        st.plotly_chart(update_fig(f, "VWAP"), use_container_width=True, key=f"vw_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('VWAP', 'N/A')}")
    with r2b:
        f = go.Figure()
        f.add_trace(go.Scatter(x=idx, y=df_c['Close'], name='Close', line=dict(color='gray', width=1)))
        f.add_trace(go.Scatter(x=idx, y=df_c['SMA_20'], name='SMA20'))
        f.add_trace(go.Scatter(x=idx, y=df_c['SMA_50'], name='SMA50'))
        f.add_trace(go.Scatter(x=idx, y=df_c['EMA_200'], name='EMA200', line=dict(color='yellow')))
        st.plotly_chart(update_fig(f, "SMA & EMA Trends"), use_container_width=True, key=f"ma_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('EMA', 'N/A')}")

    # Row 3: ATR | CCI
    r3a, r3b = st.columns(2)
    with r3a:
        f = go.Figure()
        f.add_trace(go.Scatter(x=idx, y=df_c['ATR'], name='ATR', line=dict(color='cyan')))
        st.plotly_chart(update_fig(f, "ATR (Volatility)"), use_container_width=True, key=f"at_{plotly_template}")
        st.caption("ℹ️ **Note:** High ATR = High Volatility")
    with r3b:
        f = go.Figure()
        if 'CCI' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['CCI'], name='CCI', line=dict(color='magenta')))
            f.add_hline(y=100, line_dash='dot'); f.add_hline(y=-100, line_dash='dot')
        st.plotly_chart(update_fig(f, "CCI"), use_container_width=True, key=f"cc_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('CCI', 'N/A')}")

    # Row 4: CMF | OBV
    r4a, r4b = st.columns(2)
    with r4a:
        f = go.Figure()
        if 'CMF' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['CMF'], name='CMF', fill='tozeroy'))
            f.add_hline(y=0)
        st.plotly_chart(update_fig(f, "Chaikin Money Flow"), use_container_width=True, key=f"cm_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('CMF', 'N/A')}")
    with r4b:
        f = go.Figure()
        if 'OBV' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['OBV'], name='OBV'))
        st.plotly_chart(update_fig(f, "On-Balance Volume"), use_container_width=True, key=f"ob_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('OBV', 'N/A')}")

    # Row 5: MACD | ADX
    r5a, r5b = st.columns(2)
    with r5a:
        f = go.Figure()
        if 'MACD_12_26_9' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['MACD_12_26_9'], name='MACD'))
            f.add_trace(go.Scatter(x=idx, y=df_c['MACDs_12_26_9'], name='Sig'))
            f.add_bar(x=idx, y=df_c['MACDh_12_26_9'], name='Hist')
        st.plotly_chart(update_fig(f, "MACD"), use_container_width=True, key=f"md_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('MACD', 'N/A')}")
    with r5b:
        f = go.Figure()
        if 'ADX_14' in df_c.columns:
            f.add_trace(go.Scatter(x=idx, y=df_c['ADX_14'], name='ADX', line=dict(color='white')))
            f.add_trace(go.Scatter(x=idx, y=df_c['DMP_14'], name='DI+', line=dict(color='green')))
            f.add_trace(go.Scatter(x=idx, y=df_c['DMN_14'], name='DI-', line=dict(color='red')))
        st.plotly_chart(update_fig(f, "ADX & DI"), use_container_width=True, key=f"ad_{plotly_template}")
        st.caption(f"**Interpretation:** {ct.get('ADX', 'N/A')}")

    # Row 6: ADL (Full Width)
    st.markdown("---") # Visual separator
    f = go.Figure()
    if 'ADL' in df_c.columns:
        f.add_trace(go.Scatter(x=idx, y=df_c['ADL'], name='ADL', line=dict(color='cyan')))
    st.plotly_chart(update_fig(f, "Accumulation/Distribution"), use_container_width=True, key=f"adl_{plotly_template}")
    st.caption(f"**Interpretation:** {ct.get('ADL', 'N/A')}")

    # --- INDICATOR ANALYSIS TABLE ---
    st.markdown("### ⚖️ Comprehensive Indicator Analysis")
    c_pos, c_neu, c_neg = st.columns(3)
    
    with c_pos:
        st.success("✅ POSITIVE")
        for i in interp_list['Positive']: st.write(f"• {i}")
        if not interp_list['Positive']: st.caption("None")
        
    with c_neu:
        st.warning("🟡 NEUTRAL")
        for i in interp_list['Neutral']: st.write(f"• {i}")
        if not interp_list['Neutral']: st.caption("None")
        
    with c_neg:
        st.error("❌ NEGATIVE")
        for i in interp_list['Negative']: st.write(f"• {i}")
        if not interp_list['Negative']: st.caption("None")
