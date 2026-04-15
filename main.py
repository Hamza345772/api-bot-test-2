from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import random
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict
import uvicorn
from tv_data_fetcher import TradingViewDataFetcher
from tradingview_ta import Interval

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Forex pairs only, as requested. Using FX_IDC as the default exchange for Forex.
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "AUDJPY", "EURAUD", "EURCAD", "CADCHF", "GBPCHF",
    "AUDCAD", "NZDJPY", "EURCHF", "GBPAUD"
]

fetcher = TradingViewDataFetcher()
price_data: Dict[str, List[float]] = {}
stats_store = {"total": 0, "wins": 0, "losses": 0}
active_signals: Dict[str, dict] = {}
connected_clients: List[WebSocket] = []

def is_weekend():
    # Saturday is 5, Sunday is 6
    now = datetime.now()
    return now.weekday() >= 5

async def update_market_data():
    """Background task to fetch real data from TradingView API"""
    while True:
        if is_weekend():
            await asyncio.sleep(60)
            continue
            
        for symbol in FOREX_PAIRS:
            try:
                # Fetching 1-minute interval analysis
                analysis = fetcher.get_analysis(symbol, "FX_IDC", "forex", Interval.INTERVAL_1_MINUTE)
                if analysis and 'close' in analysis.indicators:
                    price = analysis.indicators['close']
                    if symbol not in price_data:
                        price_data[symbol] = []
                    price_data[symbol].append(price)
                    if len(price_data[symbol]) > 200:
                        price_data[symbol].pop(0)
            except Exception as e:
                print(f"Error updating data for {symbol}: {e}")
        
        await asyncio.sleep(10) # Update every 10 seconds

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_market_data())

def compute_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = prices[-i] - prices[-i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def compute_ema(prices: List[float], period: int) -> float:
    if len(prices) < period:
        return prices[-1]
    k = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 4)

def compute_macd(prices: List[float]):
    if len(prices) < 26:
        return 0, 0
    ema12 = compute_ema(prices, 12)
    ema26 = compute_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = compute_ema(prices[-9:] if len(prices) >= 9 else prices, min(9, len(prices)))
    return round(macd_line, 4), round(signal_line, 4)

def compute_bollinger(prices: List[float], period: int = 20):
    if len(prices) < period:
        return prices[-1], prices[-1], prices[-1]
    recent = prices[-period:]
    mid = sum(recent) / period
    std = math.sqrt(sum((p - mid) ** 2 for p in recent) / period)
    return round(mid + 2 * std, 4), round(mid, 4), round(mid - 2 * std, 4)

def detect_candle_pattern(prices: List[float]) -> str:
    if len(prices) < 3:
        return "neutral"
    o1, c1 = prices[-3], prices[-2]
    o2, c2 = prices[-2], prices[-1]
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    if c1 < o1 and c2 > o2 and body2 > body1 * 1.5:
        return "bullish_engulfing"
    if c1 > o1 and c2 < o2 and body2 > body1 * 1.5:
        return "bearish_engulfing"
    wick = abs(prices[-1] - prices[-2])
    if wick > body2 * 2:
        return "pin_bar_up" if prices[-1] > prices[-2] else "pin_bar_down"
    return "neutral"

def detect_market_structure(prices: List[float]) -> str:
    if len(prices) < 20:
        return "ranging"
    segment = prices[-20:]
    highs = [max(segment[i:i+5]) for i in range(0, 15, 5)]
    lows = [min(segment[i:i+5]) for i in range(0, 15, 5)]
    if highs[-1] > highs[-2] > highs[-3] and lows[-1] > lows[-2]:
        return "uptrend"
    if highs[-1] < highs[-2] < highs[-3] and lows[-1] < lows[-2]:
        return "downtrend"
    return "ranging"

def detect_order_block(prices: List[float]) -> str:
    if len(prices) < 10:
        return "none"
    recent = prices[-10:]
    impulse = recent[-1] - recent[0]
    if abs(impulse) > 0.0015: # Adjusted for Forex scale
        return "bullish_ob" if impulse > 0 else "bearish_ob"
    return "none"

def detect_fvg(prices: List[float]) -> str:
    if len(prices) < 3:
        return "none"
    if prices[-3] < prices[-1] and prices[-2] > prices[-3]:
        return "bullish_fvg"
    if prices[-3] > prices[-1] and prices[-2] < prices[-3]:
        return "bearish_fvg"
    return "none"

def detect_liquidity_sweep(prices: List[float]) -> str:
    if len(prices) < 15:
        return "none"
    recent = prices[-15:]
    prev_high = max(recent[:-3])
    prev_low = min(recent[:-3])
    last = recent[-1]
    if recent[-2] > prev_high and last < prev_high:
        return "sweep_high"
    if recent[-2] < prev_low and last > prev_low:
        return "sweep_low"
    return "none"

def analyze_pair(pair: str, duration: int) -> dict:
    if is_weekend():
        return {"status": "weekend", "message": "Market is closed. It will open on Monday."}
        
    if pair not in price_data or len(price_data[pair]) < 30:
        # Fallback to direct API fetch if background task hasn't filled enough data
        try:
            analysis = fetcher.get_analysis(pair, "FX_IDC", "forex", Interval.INTERVAL_1_MINUTE)
            if not analysis: return None
            # We need historical data for these functions, so if price_data is empty, we can't do full analysis yet
            if pair not in price_data: return None
        except:
            return None

    prices = price_data[pair]
    rsi = compute_rsi(prices)
    ema5 = compute_ema(prices, 5)
    ema13 = compute_ema(prices, 13)
    ema50 = compute_ema(prices, min(50, len(prices)))
    macd_line, signal_line = compute_macd(prices)
    bb_upper, bb_mid, bb_lower = compute_bollinger(prices)
    candle = detect_candle_pattern(prices)
    structure = detect_market_structure(prices)
    order_block = detect_order_block(prices)
    fvg = detect_fvg(prices)
    liq_sweep = detect_liquidity_sweep(prices)
    current_price = prices[-1]

    bull_score = 0
    bear_score = 0
    reasons = []

    # RSI
    if rsi < 30:
        bull_score += 20
        reasons.append("RSI Oversold")
    elif rsi < 45:
        bull_score += 10
        reasons.append("RSI Bullish Zone")
    elif rsi > 70:
        bear_score += 20
        reasons.append("RSI Overbought")
    elif rsi > 55:
        bear_score += 10
        reasons.append("RSI Bearish Zone")

    # EMA
    if ema5 > ema13:
        bull_score += 15
        reasons.append("EMA Bullish Cross")
    else:
        bear_score += 15
        reasons.append("EMA Bearish Cross")

    if current_price > ema50:
        bull_score += 10
        reasons.append("Above EMA50")
    else:
        bear_score += 10
        reasons.append("Below EMA50")

    # MACD
    if macd_line > signal_line:
        bull_score += 15
        reasons.append("MACD Bullish")
    else:
        bear_score += 15
        reasons.append("MACD Bearish")

    # Bollinger
    if current_price <= bb_lower:
        bull_score += 15
        reasons.append("BB Lower Band Bounce")
    elif current_price >= bb_upper:
        bear_score += 15
        reasons.append("BB Upper Band Reversal")

    # Market Structure (SMC)
    if structure == "uptrend":
        bull_score += 15
        reasons.append("SMC: Uptrend Structure")
    elif structure == "downtrend":
        bear_score += 15
        reasons.append("SMC: Downtrend Structure")

    # Order Block (ICT)
    if order_block == "bullish_ob":
        bull_score += 20
        reasons.append("ICT: Bullish Order Block")
    elif order_block == "bearish_ob":
        bear_score += 20
        reasons.append("ICT: Bearish Order Block")

    # FVG (ICT)
    if fvg == "bullish_fvg":
        bull_score += 15
        reasons.append("ICT: Bullish FVG")
    elif fvg == "bearish_fvg":
        bear_score += 15
        reasons.append("ICT: Bearish FVG")

    # Liquidity Sweep (ICT)
    if liq_sweep == "sweep_high":
        bear_score += 20
        reasons.append("ICT: Liquidity Sweep High")
    elif liq_sweep == "sweep_low":
        bull_score += 20
        reasons.append("ICT: Liquidity Sweep Low")

    # Candle Pattern
    if candle == "bullish_engulfing":
        bull_score += 15
        reasons.append("Bullish Engulfing Candle")
    elif candle == "bearish_engulfing":
        bear_score += 15
        reasons.append("Bearish Engulfing Candle")
    elif candle == "pin_bar_up":
        bull_score += 10
        reasons.append("Pin Bar Reversal Up")
    elif candle == "pin_bar_down":
        bear_score += 10
        reasons.append("Pin Bar Reversal Down")

    total = bull_score + bear_score
    if total == 0:
        return None

    if bull_score > bear_score:
        direction = "UP"
        raw_conf = bull_score / total
    else:
        direction = "DOWN"
        raw_conf = bear_score / total

    signals_count = len(reasons)
    if signals_count < 3:
        return None

    confidence = round(50 + raw_conf * 48, 1)
    confidence = min(98, max(52, confidence))
    probability_score = round(confidence + (signals_count * 0.5), 2)

    now = datetime.now()
    entry_in = random.randint(5, 25)
    entry_time = datetime.fromtimestamp(now.timestamp() + entry_in)

    return {
        "pair": pair,
        "direction": direction,
        "confidence": confidence,
        "probability_score": probability_score,
        "reasons": reasons[:5],
        "signals_count": signals_count,
        "rsi": rsi,
        "structure": structure,
        "entry_in": entry_in,
        "entry_time": entry_time.strftime("%H:%M:%S"),
        "duration": duration,
        "status": "ready",
        "id": f"{pair}-{int(now.timestamp())}",
        "timestamp": now.isoformat()
    }

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/pairs")
async def get_pairs():
    if is_weekend():
        return {"pairs": [], "status": "weekend", "message": "Market is closed. It will open on Monday."}
    return {"pairs": FOREX_PAIRS, "status": "open"}

@app.get("/api/stats")
async def get_stats():
    total = stats_store["total"]
    wins = stats_store["wins"]
    acc = round((wins / total * 100), 1) if total > 0 else 0
    return {**stats_store, "accuracy": acc}

@app.post("/api/result")
async def post_result(data: dict):
    res = data.get("result")
    if res == "win":
        stats_store["wins"] += 1
    elif res == "loss":
        stats_store["losses"] += 1
    stats_store["total"] += 1
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if is_weekend():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Market is closed. It will open on Monday."
                }))
                continue

            if msg.get("type") == "analyze":
                pair = msg.get("pair")
                duration = msg.get("duration", 1)
                result = analyze_pair(pair, duration)
                if result:
                    await websocket.send_text(json.dumps({
                        "type": "signal",
                        "data": result
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Insufficient data for {pair}. Please wait a few moments."
                    }))
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
