import os
import warnings
import yfinance as yf
import pandas as pd
from datetime import datetime # <--- NEW IMPORT
from crewai import Agent, Task, Crew, Process

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from crewai.tools import BaseTool
except ImportError:
    from crewai_tools import BaseTool

from ddgs import DDGS

# ==============================================================================
# CONFIGURATION
# ==============================================================================

os.environ["OPENAI_API_KEY"] = "add_your_key" # Enter your key here

# ==============================================================================
# TOOL 1: PRICE FETCHER
# ==============================================================================

class XAUPriceFetchTool(BaseTool):
    name: str = "XAUUSD 4H Price Fetcher"
    description: str = "Fetches Gold (XAU/USD) market data with RSI and EMA indicators."

    def _run(self, query: str) -> str:
        try:
            print(f"\n[Tool Log] Fetching 4H Price Data from Yahoo Finance...") 
            ticker = "GC=F" 
            data = yf.Ticker(ticker)
            history = data.history(period="1mo", interval="1h")
            if history.empty:
                return "Error: Could not fetch data."

            ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
            data_4h = history.resample('4h').agg(ohlc_dict).dropna()

            # Indicators
            delta = data_4h['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data_4h['RSI'] = 100 - (100 / (1 + rs))
            data_4h['EMA_50'] = data_4h['Close'].ewm(span=50, adjust=False).mean()

            final_df = data_4h[['Close', 'RSI', 'EMA_50']].tail(12).round(2)
            return final_df.to_string()

        except Exception as e:
            return f"Error processing data: {str(e)}"

# ==============================================================================
# TOOL 2: NEWS SEARCH (TIMELIMIT FIX)
# ==============================================================================

class NewsSearchTool(BaseTool):
    name: str = "DuckDuckGo News Search"
    description: str = "Search the web for current events and market sentiment."

    def _run(self, query: str) -> str:
        try:
            print(f"\n[Tool Log] Searching DuckDuckGo (Past 24h) for: '{query}'...") 
            results = []
            
            with DDGS() as ddgs:
                # --- THE FIX ---
                # timelimit='d' forces results from the past Day.
                # timelimit='w' would be past Week.
                search_gen = ddgs.text(
                    query, 
                    max_results=3, 
                    timelimit='d' # <--- CRITICAL FIX FOR FRESH NEWS
                )
                
                for r in search_gen:
                    title = r.get('title', 'No Title')
                    body = r.get('body', 'No snippet')
                    date = r.get('date', 'Unknown Date') # Try to get the date tag
                    results.append(f"NEWS TITLE: {title}\nDATE: {date}\nSNIPPET: {body}\n---")
            
            if not results:
                return "No news found in the last 24 hours. Assume Neutral."
                
            return "\n".join(results)

        except Exception as e:
            return f"Search Error: {str(e)}"

# Instantiate
price_tool = XAUPriceFetchTool()
search_tool = NewsSearchTool() 

# Get Today's Date for the Prompt
current_date = datetime.now().strftime("%Y-%m-%d")

# ==============================================================================
# AGENTS
# ==============================================================================

research_manager = Agent(
    role='Senior Market Research Manager',
    goal=f'Determine the fundamental bias for Gold as of {current_date}.',
    backstory=f"""You are a veteran macro-economist. 
    Today is {current_date}.
    You STRICTLY check the dates of news articles. 
    If an article is not from {current_date} or yesterday, IGNORE IT.
    You analyze the Fed, Inflation, and Geopolitics.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

technical_analyst = Agent(
    role='Lead Technical Analyst',
    goal='Analyze 4-Hour charts with Indicators to provide a Trade Setup.',
    backstory="""You are a Quant Swing Trader.
    Strategy:
    1. Check Manager's Bias.
    2. Trend: Price > EMA_50 (Up/Buy), Price < EMA_50 (Down/Sell).
    3. Momentum: RSI > 70 (Overbought), RSI < 30 (Oversold).
    
    Make a decision based on the DATA provided by the tool.""",
    verbose=True,
    allow_delegation=False,
    tools=[price_tool]
)

# ==============================================================================
# TASKS
# ==============================================================================

research_task = Task(
    description=f"""
    1. Search for "Gold Price News {current_date}" and "Federal Reserve outlook".
    2. Verify the news is actually recent (from the last 24-48 hours).
    3. Decide: Is the Fundamental Sentiment BULLISH, BEARISH, or NEUTRAL?
    """,
    expected_output='A brief Market Sentiment Report based ONLY on fresh news.',
    agent=research_manager
)

analysis_task = Task(
    description="""
    1. Get the latest price and indicators.
    2. Combine with Manager's sentiment.
    3. Output the Trade Signal (Action, Entry, SL, TP).
    """,
    expected_output="Final Trade Signal with Reasoning.",
    agent=technical_analyst,
    context=[research_task] 
)

# ==============================================================================
# EXECUTION
# ==============================================================================

xau_copilot_crew = Crew(
    agents=[research_manager, technical_analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("################################################")
    print(f"## STARTING XAUCOPILOT (DATE: {current_date}) ... ##")
    print("################################################\n")
    
    result = xau_copilot_crew.kickoff()
    
    print("\n\n################################################")
    print("## FINAL RECOMMENDATION ##")
    print("################################################\n")
    print(result)
