## FinFusion - AI-Powered Portfolio Optimization Agent 📈

FinFusion is an intelligent investment advisory platform that combines LangChain agents, real-time market data, and historical analysis to provide personalized portfolio recommendations and stock insights.


Features
1. AI Chat Agent 
- Conversational AI advisor powered by OpenAI GPT or Google Gemini
- Detailed stock analysis with facts and metrics
- Memory-enabled conversations for contextual understanding
- Real-time price data and historical performance analysis

2. Smart Portfolio Generation 
- Risk-based portfolio recommendations (Low/Medium/High)
- Automatic stock selection based on volatility and returns
- Current price integration via Finnhub API
- AI-generated investment analysis and rationale
- Customizable investment amounts and stock counts

3. Market Data Integration 
- Live S&P 500 chart via TradingView
- Real-time stock quotes from Finnhub
- Historical performance metrics
- News sentiment analysis (positive/negative/neutral)

4. Interactive Dashboard 
- Top performing stocks visualization
- Volatility analysis charts
- Company name mapping for better readability
- Portfolio metrics and statistics


Key Components
- LangChain Agents: Orchestrates tool usage and reasoning
- Custom Tools: 3 optimized tools for stock analysis
- Memory System: Maintains conversation context (last 4 exchanges)

Data Sources:
- Historical: CSV files (quantitative_summary.csv, final_news.csv, top_100_sp500.csv)
- Real-time: Finnhub API
- Market Data: TradingView widgets


Tools Available
1. AnalyzeStock
pythonInput: Ticker symbol (e.g., "AAPL")
Output: Current price, historical performance, sentiment analysis
2. GetTopStocks
pythonInput: "return" | "stable" | "positive_sentiment"
Output: Top 10 stocks by selected criteria
3. EvaluatePortfolio
pythonInput: Comma-separated tickers (e.g., "AAPL,MSFT,GOOGL")
Output: Portfolio metrics and analysis

⚙️ Configuration Options
LLM Settings
- OpenAI Models: gpt-4o-mini (recommended), gpt-3.5-turbo, gpt-4o
- Gemini Models: gemini-1.5-flash (recommended), gemini-2.0-flash-exp, gemini-1.5-pro
- Temperature: 0.2 (natural responses)
- Max Tokens: 8000 (detailed responses)

Agent Settings
- Memory Window: 4 conversations
- Max Iterations: 5 tool calls per query
- Error Handling: Automatic fallback to direct data lookup

Key Features Explained
Risk-Based Stock Selection
- Low Risk: Selects stocks with lowest volatility
- Medium Risk: Balances return-to-volatility ratio
- High Risk: Selects stocks with highest returns


Memory System
- Remembers last 4 conversation exchanges
- Enables follow-up questions
- Maintains context across queries


Performance
- Response Time: 2-5 seconds per query
- Portfolio Generation: 5-8 seconds
- Token Usage: ~500-800 tokens per detailed response
- Cost per Query: ~$0.001-$0.003 (with gpt-4o-mini)


Future Enhancements
- Add more LLM providers (Anthropic Claude, Cohere)
- Real-time portfolio tracking
- Export portfolio to PDF
- Multi-language support
- Advanced charting with technical indicators

⚠️ Disclaimer
This application is for educational and informational purposes only. It does not constitute financial advice, investment advice, trading advice, or any other sort of advice. You should not treat any of the application's content as such. Past performance does not guarantee future results. Always do your own research and consult with a qualified financial advisor before making investment decisions.
