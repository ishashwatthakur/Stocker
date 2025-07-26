from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from typing import Optional
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache setup
_cache = {}
CACHE_TTL = 3600  # 1 hour

def set_cache(key: str, data: dict):
    _cache[key] = {
        "timestamp": datetime.datetime.now(),
        "data": data
    }

def get_cache(key: str):
    if key in _cache:
        if (datetime.datetime.now() - _cache[key]["timestamp"]).seconds < CACHE_TTL:
            return _cache[key]["data"]
    return None

# Enhanced NSE Scraper
def scrape_nse_data(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        # First get cookies
        session = requests.Session()
        session.get(f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}", 
                   headers=headers, timeout=10)
        
        # Fetch historical data
        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={start_date.strftime('%d-%m-%Y')}&to={end_date.strftime('%d-%m-%Y')}"
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        if not data.get('data'):
            return None
            
        df = pd.DataFrame(data['data'])
        df = df.rename(columns={
            'CH_TIMESTAMP': 'Date',
            'CH_OPENING_PRICE': 'Open',
            'CH_TRADE_HIGH_PRICE': 'High',
            'CH_TRADE_LOW_PRICE': 'Low',
            'CH_CLOSING_PRICE': 'Close',
            'CH_TOTAL_TRADED_QTY': 'Volume'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.sort_values('Date')
        
    except Exception as e:
        logger.error(f"NSE scraping failed for {symbol}: {str(e)}")
        return None

# Technical Indicators
def calculate_indicators(df: pd.DataFrame):
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.replace({np.nan: None})

# Main API Endpoint
@app.get("/api/stock-data/{symbol}")
async def get_stock_data(symbol: str, start_date: str, end_date: str):
    cache_key = f"{symbol}_{start_date}_{end_date}"
    if cached := get_cache(cache_key):
        return cached
    
    try:
        # Clean symbol
        symbol = symbol.replace('.NS', '').replace('.BO', '').upper()
        if not symbol.isalpha():
            raise HTTPException(400, "Invalid symbol format")
            
        # Parse dates
        try:
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(400, "Invalid date format (use YYYY-MM-DD)")
            
        # Try multiple data sources
        hist_df = None
        info = {}
        
        # 1. Try yfinance with both NSE and BSE suffixes
        for suffix in ['', '.NS', '.BO']:
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                data = ticker.history(
                    start=start_dt,
                    end=end_dt + datetime.timedelta(days=1),
                    auto_adjust=True,
                    timeout=10
                )
                if not data.empty:
                    hist_df = data
                    info = ticker.info
                    break
            except Exception as e:
                logger.warning(f"yfinance failed for {symbol}{suffix}: {str(e)}")
                continue
                
        # 2. Fallback to NSE scraping
        if hist_df is None or hist_df.empty:
            hist_df = scrape_nse_data(symbol, start_dt, end_dt)
            
        # 3. Final fallback to last available data
        if hist_df is None or hist_df.empty:
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist_df = ticker.history(period="1y")
                info = ticker.info
            except:
                pass
                
        if hist_df is None or hist_df.empty:
            raise HTTPException(404, f"No data found for {symbol}")
            
        # Process data
        hist_df = hist_df.reset_index()
        if 'Date' not in hist_df.columns:
            hist_df = hist_df.rename(columns={'index': 'Date'})
            
        hist_df = calculate_indicators(hist_df)
        
        # Format response
        result = {
            "history": hist_df.to_dict('records'),
            "info": info
        }
        set_cache(cache_key, result)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(500, "Internal server error")

# Other endpoints
@app.get("/api/symbols")
async def get_symbols():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        return {"symbols": sorted(df['SYMBOL'].unique().tolist())}
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch symbols: {str(e)}")

# Static files
app.mount("/", StaticFiles(directory="public", html=True), name="static")