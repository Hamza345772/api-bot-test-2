import time
from pytradingview import TVclient
from tradingview_ta import TA_Handler, Interval, Exchange

class TradingViewDataFetcher:
    """
    A class to fetch real-time and technical analysis data from TradingView.
    Supports Forex, Crypto, and Stocks.
    """

    def __init__(self, username=None, password=None):
        self.client = TVclient(username=username, password=password) if username and password else TVclient()
        self.chart = self.client.chart
        self.latest_price = None
        self.symbol_info = None

    def get_analysis(self, symbol, exchange, screener, interval=Interval.INTERVAL_1_MINUTE):
        """
        Fetches technical analysis indicators for a given symbol.
        :param symbol: e.g., 'BTCUSDT'
        :param exchange: e.g., 'BINANCE'
        :param screener: e.g., 'crypto' or 'forex'
        :param interval: Timeframe interval (e.g., Interval.INTERVAL_1_MINUTE)
        :return: Analysis object containing indicators and summary.
        """
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener=screener,
            interval=interval
        )
        try:
            analysis = handler.get_analysis()
            return analysis
        except Exception as e:
            print(f"Error fetching analysis for {symbol}: {e}")
            return None

    def start_realtime_stream(self, symbol_with_exchange, timeframe="1"):
        """
        Starts a WebSocket connection to stream real-time price data.
        :param symbol_with_exchange: e.g., 'BINANCE:BTCUSDT' or 'FX:EURUSD'
        :param timeframe: Chart timeframe (e.g., '1' for 1 minute)
        """
        self.chart.set_up_chart()
        self.chart.set_market(symbol_with_exchange, {"timeframe": timeframe})

        def on_loaded(_):
            self.symbol_info = self.chart.get_infos
            print(f"✅ Market Loaded: {self.symbol_info.get('description', symbol_with_exchange)}")

        def on_update(_):
            periods = self.chart.get_periods
            if periods:
                self.latest_price = periods.get('close')
                print(f"🟢 {symbol_with_exchange} New Price: {self.latest_price}")

        self.chart.on_symbol_loaded(on_loaded)
        self.chart.on_update(on_update)

        print(f"🚀 Starting WebSocket for {symbol_with_exchange}...")
        self.client.create_connection()

if __name__ == "__main__":
    # Example Usage
    fetcher = TradingViewDataFetcher()

    # 1. Fetch Technical Analysis for Crypto
    print("\n--- Crypto Analysis (BTCUSDT) ---")
    btc_analysis = fetcher.get_analysis("BTCUSDT", "BINANCE", "crypto")
    if btc_analysis:
        print(f"Summary: {btc_analysis.summary}")
        print(f"RSI: {btc_analysis.indicators.get('RSI')}")
        print(f"Current Close: {btc_analysis.indicators.get('close')}")

    # 2. Fetch Technical Analysis for Forex
    print("\n--- Forex Analysis (EURUSD) ---")
    eur_analysis = fetcher.get_analysis("EURUSD", "FX_IDC", "forex")
    if eur_analysis:
        print(f"Summary: {eur_analysis.summary}")
        print(f"Current Close: {eur_analysis.indicators.get('close')}")

    # 3. Start Real-time Stream (Optional - will block execution)
    # print("\n--- Starting Real-time Stream for BTCUSDT ---")
    # fetcher.start_realtime_stream("BINANCE:BTCUSDT")
