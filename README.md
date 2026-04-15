# Quotex OTC Signal Bot

Professional binary options signal bot for Quotex OTC market with SMC + ICT + RSI strategy engine.

## Features
- Real-time scanner for all 48 OTC pairs
- SMC (Smart Money Concepts) analysis
- ICT (Inner Circle Trader) — Order Blocks, FVG, Liquidity Sweeps
- RSI, EMA, MACD, Bollinger Bands
- Trade status: Ready → Active → WIN/LOSS
- Entry countdown timer
- Candle info — exactly when to enter
- Signals sorted by highest probability first
- Overall accuracy dashboard
- Fully responsive — mobile + desktop

## Deploy on Railway (FREE) — Step by Step

### Step 1: Upload to GitHub
1. Go to github.com → Create new repository → Name it `quotex-bot`
2. Upload ALL files from this folder:
   - main.py
   - requirements.txt
   - Dockerfile
   - railway.toml
   - static/index.html (create static folder first)

### Step 2: Deploy on Railway
1. Go to railway.app → Login with GitHub
2. Click "New Project"
3. Click "Deploy from GitHub repo"
4. Select your `quotex-bot` repository
5. Railway will auto-detect Dockerfile and deploy!

### Step 3: Get Your Live Link
1. After deploy, click your project
2. Go to "Settings" → "Networking"
3. Click "Generate Domain"
4. Your live link is ready!

Example: `https://quotex-bot-production.up.railway.app`

## Local Testing (Optional)
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open: http://localhost:8000

## Strategy Engine
Signals are generated using:
- **RSI** — Oversold/Overbought detection
- **EMA 5/13/50** — Trend direction + crossovers
- **MACD** — Momentum confirmation
- **Bollinger Bands** — Volatility + reversal zones
- **SMC Market Structure** — HH/HL/LH/LL detection
- **ICT Order Blocks** — Institutional entry zones
- **ICT Fair Value Gaps (FVG)** — Price imbalances
- **ICT Liquidity Sweeps** — Stop hunt detection
- **Candle Patterns** — Engulfing, Pin Bar confirmation

Minimum 3 confluences required for a signal to appear.
Signals sorted by probability score (highest first).
