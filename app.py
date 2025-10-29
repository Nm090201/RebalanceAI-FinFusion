import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import finnhub
from datetime import datetime
import plotly.express as px
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Page config
st.set_page_config(page_title="FinFusion", page_icon="📈", layout="wide")

# CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">💰FinFusion : AI-Powered Investment Portfolio Analyzer</h1>', unsafe_allow_html=True)

# ===========================
# SESSION STATE INITIALIZATION
# ===========================
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'data_store' not in st.session_state:
    st.session_state.data_store = {'quant_df': None}
if 'latest_portfolio' not in st.session_state:
    st.session_state.latest_portfolio = None
if 'generate_portfolio' not in st.session_state:
    st.session_state.generate_portfolio = False
if 'portfolio_params' not in st.session_state:
    st.session_state.portfolio_params = None

# ===========================
# LOAD DATA (Cached)
# ===========================
@st.cache_data
def load_data():
    news_df = pd.read_csv('final_news.csv')
    quant_df = pd.read_csv('quantitative_summary.csv')
    return news_df, quant_df

news_df, quant_df = load_data()
st.session_state.data_store['quant_df'] = quant_df

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ===========================
# ENHANCED TOOL FUNCTIONS
# ===========================

def analyze_stock(ticker: str) -> str:
    """Get complete analysis: current price + historical data + sentiment"""
    result = []
    ticker = ticker.upper().strip()
    
    # Current price
    try:
        quote = finnhub_client.quote(ticker)
        result.append(f"CURRENT PRICE: ${quote['c']:.2f}")
        result.append(f"TODAY'S CHANGE: {quote['dp']:+.2f}% (${quote['d']:+.2f})")
        result.append(f"DAY RANGE: ${quote['l']:.2f} - ${quote['h']:.2f}")
        result.append(f"PREVIOUS CLOSE: ${quote['pc']:.2f}")
    except:
        result.append("Current price: Not available")
    
    # Historical data
    stock_data = quant_df[quant_df['Stock'] == ticker]
    if not stock_data.empty:
        data = stock_data.iloc[0]
        result.append(f"\nHISTORICAL PERFORMANCE:")
        result.append(f"- Cumulative Return: {data['Cumulative Return (%)']:.2f}%")
        result.append(f"- Average Daily Return: {data['Mean Daily Return (%)']:.4f}%")
        result.append(f"- Volatility: {data['Volatility (%)']:.2f}%")
        result.append(f"- Average Price: ${data['Mean Close']:.2f}")
        result.append(f"- Price Range: ${data['Min Close']:.2f} - ${data['Max Close']:.2f}")
    else:
        result.append("\nHistorical data: Not available in dataset")
    
    # Sentiment
    stock_news = news_df[news_df['Ticker'] == ticker]
    if not stock_news.empty:
        pos = (stock_news['sentiment'] == 'positive').sum()
        neg = (stock_news['sentiment'] == 'negative').sum()
        neu = (stock_news['sentiment'] == 'neutral').sum()
        total = len(stock_news)
        avg_conf = stock_news['confidence'].mean()
        
        result.append(f"\nNEWS SENTIMENT:")
        result.append(f"- Total articles analyzed: {total}")
        result.append(f"- Positive: {pos} ({pos/total*100:.1f}%)")
        result.append(f"- Negative: {neg} ({neg/total*100:.1f}%)")
        result.append(f"- Neutral: {neu} ({neu/total*100:.1f}%)")
        result.append(f"- Average confidence: {avg_conf:.2f}")
        
        # Recent headlines
        recent = stock_news.sort_values('Date', ascending=False).head(3)['Headline'].tolist()
        if recent:
            result.append(f"\nRECENT HEADLINES:")
            for i, headline in enumerate(recent[:3], 1):
                result.append(f"{i}. {headline[:80]}...")
    else:
        result.append("\nSentiment data: Not available")
    
    return "\n".join(result)

def get_top_stocks(criteria: str) -> str:
    """Get top stocks by: 'return', 'stable', or 'positive_sentiment'"""
    if criteria == "return":
        top = quant_df.nlargest(10, 'Cumulative Return (%)')
        result = "TOP 10 STOCKS BY RETURN:\n\n"
        for i, (_, row) in enumerate(top.iterrows(), 1):
            result += f"{i}. {row['Stock']}: {row['Cumulative Return (%)']:.2f}% return\n"
            result += f"   Volatility: {row['Volatility (%)']:.2f}% | Avg Price: ${row['Mean Close']:.2f}\n\n"
        return result
    
    elif criteria == "stable":
        top = quant_df.nsmallest(10, 'Volatility (%)')
        result = "TOP 10 LOW VOLATILITY STOCKS:\n\n"
        for i, (_, row) in enumerate(top.iterrows(), 1):
            result += f"{i}. {row['Stock']}: {row['Volatility (%)']:.2f}% volatility\n"
            result += f"   Return: {row['Cumulative Return (%)']:.2f}% | Avg Price: ${row['Mean Close']:.2f}\n\n"
        return result
    
    elif criteria == "positive_sentiment":
        sentiment_summary = news_df[news_df['sentiment'] == 'positive'].groupby('Ticker').agg({
            'sentiment': 'count',
            'confidence': 'mean'
        }).reset_index()
        sentiment_summary.columns = ['Ticker', 'positive_count', 'avg_confidence']
        top = sentiment_summary.nlargest(10, 'positive_count')
        
        result = "TOP 10 STOCKS WITH POSITIVE SENTIMENT:\n\n"
        for i, (_, row) in enumerate(top.iterrows(), 1):
            ticker = row['Ticker']
            stock_data = quant_df[quant_df['Stock'] == ticker]
            perf = ""
            if not stock_data.empty:
                perf = f" | Return: {stock_data.iloc[0]['Cumulative Return (%)']:.2f}%"
            
            result += f"{i}. {ticker}: {row['positive_count']} positive articles\n"
            result += f"   Confidence: {row['avg_confidence']:.2f}{perf}\n\n"
        return result
    
    return "Invalid criteria. Use: 'return', 'stable', or 'positive_sentiment'"

def calculate_portfolio(tickers: str) -> str:
    """Evaluate portfolio of comma-separated tickers"""
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    portfolio_data = quant_df[quant_df['Stock'].isin(ticker_list)]
    
    if portfolio_data.empty:
        return "No valid tickers found in the dataset"
    
    avg_return = portfolio_data['Cumulative Return (%)'].mean()
    avg_vol = portfolio_data['Volatility (%)'].mean()
    total_stocks = len(portfolio_data)
    best_performer = portfolio_data.loc[portfolio_data['Cumulative Return (%)'].idxmax()]
    
    result = f"PORTFOLIO ANALYSIS ({total_stocks} stocks):\n\n"
    result += f"STOCKS: {', '.join(portfolio_data['Stock'].tolist())}\n\n"
    result += f"METRICS:\n"
    result += f"- Average Return: {avg_return:.2f}%\n"
    result += f"- Average Volatility: {avg_vol:.2f}%\n"
    result += f"- Risk Level: {'Low' if avg_vol < 2 else 'High' if avg_vol > 3 else 'Medium'}\n"
    result += f"- Best Performer: {best_performer['Stock']} ({best_performer['Cumulative Return (%)']:.2f}%)\n\n"
    
    result += "INDIVIDUAL STOCKS:\n"
    for _, stock in portfolio_data.iterrows():
        result += f"- {stock['Stock']}: {stock['Cumulative Return (%)']:.2f}% return, {stock['Volatility (%)']:.2f}% volatility\n"
    
    return result
        
# ===========================
# TOOL SET
# ===========================
tools = [
    Tool(
        name="AnalyzeStock",
        func=analyze_stock,
        description="Get detailed stock analysis including current price, historical performance, and sentiment. Input: ticker symbol like 'AAPL'"
    ),
    Tool(
        name="GetTopStocks",
        func=get_top_stocks,
        description="Get top 10 stocks by criteria. Input: 'return' for highest returns, 'stable' for lowest volatility, or 'positive_sentiment' for best news"
    ),
    Tool(
        name="EvaluatePortfolio",
        func=calculate_portfolio,
        description="Analyze a portfolio of stocks. Input: comma-separated tickers like 'AAPL,MSFT,GOOGL'"
    )
]

# ===========================
# AGENT WITH RESPONSES
# ===========================
def initialize_agent_custom(llm_provider, api_key, model_name, news_df, quant_df):
    """Initialize agent with improved response quality"""
    try:
        if llm_provider == "OpenAI":
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.2,  
                openai_api_key=api_key,
                max_tokens=800  
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                google_api_key=api_key,
                max_output_tokens=800
            )
        
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=4,  
            return_messages=False
        )
        st.session_state.memory = memory
        
        prompt = PromptTemplate.from_template("""You're an expert financial advisor. Provide detailed, informative answers with facts and context.

IMPORTANT INSTRUCTIONS:
- Give comprehensive answers with relevant facts and data
- Use the tool results to provide specific numbers and metrics
- Explain trends, comparisons, and implications
- Format your answers clearly with bullet points or sections when appropriate
- Always provide context and reasoning, not just raw data
- If analyzing stocks, mention key metrics like returns, volatility, and sentiment

Available Tools:
{tools}

Tool Names: {tool_names}

Format:
Question: the user's question
Thought: what information do I need to provide a comprehensive answer
Action: tool to use
Action Input: tool input
Observation: tool result
Thought: I now have data, let me provide a detailed, informative answer
Final Answer: [Detailed answer with facts, context, and insights]

Conversation History:
{chat_history}

Question: {input}
{agent_scratchpad}""")
        
        agent = create_react_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False
        )
        
        return agent_executor
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# ===========================
# MAIN UI
# ===========================
tab1, tab2, tab3 = st.tabs(["💬 AI Agent","📈 Chart", "📊 Dashboard"])

# TAB 1: AI AGENT
with tab1:
    st.markdown("## 🤖 AI Portfolio Advisor")
    
    # Check if portfolio generation was triggered from sidebar
    if st.session_state.generate_portfolio and st.session_state.portfolio_params:
        params = st.session_state.portfolio_params
        investment_amount = params['investment_amount']
        num_stocks = params['num_stocks']
        risk_level = params['risk_level']
        
        # Reset flag immediately
        st.session_state.generate_portfolio = False
        
        with st.spinner("🔄 Generating your personalized portfolio..."):
            try:
                # Get data using Python
                if risk_level == "Low":
                    selected_stocks = quant_df.nsmallest(num_stocks, 'Volatility (%)')
                    strategy = "low volatility"
                elif risk_level == "High":
                    selected_stocks = quant_df.nlargest(num_stocks, 'Cumulative Return (%)')
                    strategy = "high return"
                else:
                    quant_df_temp = quant_df.copy()
                    quant_df_temp['score'] = quant_df_temp['Cumulative Return (%)'] / (quant_df_temp['Volatility (%)'] + 1)
                    selected_stocks = quant_df_temp.nlargest(num_stocks, 'score')
                    strategy = "balanced"
                
                allocation = investment_amount / num_stocks
                stock_details = []
                
                for _, stock in selected_stocks.iterrows():
                    ticker = stock['Stock']
                    details = {
                        'ticker': ticker,
                        'allocation': allocation,
                        'return': stock['Cumulative Return (%)'],
                        'volatility': stock['Volatility (%)'],
                        'price': 'N/A',
                        'change': 'N/A'
                    }
                    
                    try:
                        quote = finnhub_client.quote(ticker)
                        details['price'] = f"${quote['c']:.2f}"
                        details['change'] = f"{quote['dp']:+.2f}%"
                    except:
                        pass
                    
                    stock_news = news_df[news_df['Ticker'] == ticker]
                    if not stock_news.empty:
                        pos = (stock_news['sentiment'] == 'positive').sum()
                        neg = (stock_news['sentiment'] == 'negative').sum()
                        details['sentiment'] = f"{pos} positive, {neg} negative"
                    else:
                        details['sentiment'] = "No data"
                    
                    stock_details.append(details)
                
                # Use LLM for analysis
                llm_prompt = f"""Create a professional portfolio recommendation analysis.

**Portfolio:**
- Investment: ${investment_amount:,}
- Stocks: {num_stocks}
- Risk: {risk_level}
- Strategy: {strategy}

**Selected Stocks:**
"""
                for i, stock in enumerate(stock_details, 1):
                    llm_prompt += f"\n{i}. {stock['ticker']} - ${stock['allocation']:,.0f}"
                    llm_prompt += f"\n   Current: {stock['price']} ({stock['change']} today)"
                    llm_prompt += f"\n   Historical: {stock['return']:.1f}% return, {stock['volatility']:.1f}% vol"
                    llm_prompt += f"\n   Sentiment: {stock['sentiment']}"
                
                llm_prompt += f"""

Provide analysis with:
1. Why these stocks fit {risk_level} risk
2. Key portfolio strengths
3. Overall recommendation

Be detailed and informative (200-250 words)."""

                if 'agent' in st.session_state and st.session_state.agent:
                    from langchain_openai import ChatOpenAI
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    
                    if 'llm_provider' in st.session_state:
                        llm_provider = st.session_state.llm_provider
                        selected_model = st.session_state.selected_model
                        api_key = st.session_state.api_key
                        
                        if llm_provider == "OpenAI":
                            llm = ChatOpenAI(model=selected_model, temperature=0.7, openai_api_key=api_key)
                        else:
                            llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.7, google_api_key=api_key)
                        
                        llm_analysis = llm.invoke(llm_prompt).content
                    else:
                        llm_analysis = "AI analysis unavailable."
                else:
                    llm_analysis = "Please initialize the agent first."
                
                # Format output
                result_text = f"# {risk_level} Risk Portfolio\n\n"
                result_text += f"**Investment:** ${investment_amount:,} | **Stocks:** {num_stocks} | **Strategy:** {strategy.title()}\n\n"
                result_text += "---\n\n"
                result_text += "## 📊 Recommended Allocation\n\n"
                
                for i, stock in enumerate(stock_details, 1):
                    result_text += f"### {i}. {stock['ticker']} - ${stock['allocation']:,.0f} ({100/num_stocks:.1f}%)\n"
                    result_text += f"- **Current Price:** {stock['price']} ({stock['change']} today)\n"
                    result_text += f"- **Historical Performance:** {stock['return']:.1f}% return, {stock['volatility']:.1f}% volatility\n"
                    result_text += f"- **News Sentiment:** {stock['sentiment']}\n\n"
                
                result_text += "---\n\n"
                result_text += "## 🤖 AI Analysis\n\n"
                result_text += llm_analysis
                result_text += "\n\n---\n\n"
                result_text += "*This combines quantitative analysis with AI insights. Past performance doesn't guarantee future results.*"
                
                query = f"Portfolio: {num_stocks} stocks, {risk_level} risk, ${investment_amount}"
                st.session_state.latest_portfolio = {
                    'query': query,
                    'result': result_text,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.success("✅ Portfolio generated with AI analysis!")
                st.markdown(result_text)
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Show latest portfolio if exists
    if st.session_state.latest_portfolio and not st.session_state.get('generate_portfolio', False):
        with st.expander("🎯 View Latest Portfolio Recommendation", expanded=False):
            portfolio_info = st.session_state.latest_portfolio
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"*Generated: {portfolio_info['timestamp']}*")
            with col_b:
                if st.button("🗑️ Clear", key="clear_portfolio"):
                    st.session_state.latest_portfolio = None
                    st.rerun()
            
            st.markdown("---")
            st.markdown(portfolio_info['result'])
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏆 Top Performers", use_container_width=True):
            top = quant_df.nlargest(5, 'Cumulative Return (%)')
            response = "**Top 5 Performing Stocks:**\n\n"
            for idx, (_, row) in enumerate(top.iterrows(), 1):
                response += f"{idx}. **{row['Stock']}** - {row['Cumulative Return (%)']:.1f}% return, {row['Volatility (%)']:.1f}% volatility\n"
            
            st.session_state.chat_history.append(("user", "Show top 5 stocks"))
            st.session_state.chat_history.append(("assistant", response))
            st.rerun()
    
    with col2:
        if st.button("🛡️ Safe Picks", use_container_width=True):
            safe = quant_df.nsmallest(5, 'Volatility (%)')
            response = "**Top 5 Low Volatility Stocks:**\n\n"
            for idx, (_, row) in enumerate(safe.iterrows(), 1):
                response += f"{idx}. **{row['Stock']}** - {row['Volatility (%)']:.1f}% volatility, {row['Cumulative Return (%)']:.1f}% return\n"
            
            st.session_state.chat_history.append(("user", "Safe stocks"))
            st.session_state.chat_history.append(("assistant", response))
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("### 💬 Chat with Your Advisor")
    st.caption("Ask detailed questions about stocks, comparisons, or investment strategies")
    
    if not st.session_state.chat_history:
        st.info("👋 Ask me detailed questions! Try: 'Tell me about AAPL', 'Compare MSFT and GOOGL', or 'Show me tech stocks with positive sentiment'")
    
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
    
    if prompt := st.chat_input("Ask about stocks..."):
        if not st.session_state.agent:
            st.warning("⚠️ Initialize agent in sidebar first!")
        else:
            st.session_state.chat_history.append(("user", prompt))
            
            with st.spinner("🔍 Analyzing..."):
                try:
                    result = st.session_state.agent.invoke({"input": prompt})
                    response = result['output']
                    st.session_state.chat_history.append(("assistant", response))
                    st.rerun()
                except Exception as e:
                    if "iteration" in str(e).lower() or "limit" in str(e).lower():
                        fallback_response = "I encountered complexity. Here's what I found:\n\n"
                        
                        potential_ticker = prompt.upper().replace("TELL ME ABOUT", "").replace("ANALYZE", "").replace("WHAT ABOUT", "").strip().split()[0] if prompt else ""
                        
                        if potential_ticker and potential_ticker in quant_df['Stock'].values:
                            stock_data = quant_df[quant_df['Stock'] == potential_ticker].iloc[0]
                            fallback_response += f"**{potential_ticker}:**\n"
                            fallback_response += f"- Historical Return: {stock_data['Cumulative Return (%)']:.1f}%\n"
                            fallback_response += f"- Volatility: {stock_data['Volatility (%)']:.1f}%\n"
                            
                            try:
                                quote = finnhub_client.quote(potential_ticker)
                                fallback_response += f"- Current: ${quote['c']:.2f} ({quote['dp']:+.2f}%)\n"
                            except:
                                pass
                        else:
                            fallback_response += "Try: 'Analyze AAPL', 'Top performing stocks', 'Low volatility stocks'"
                        
                        response = fallback_response
                    else:
                        response = f"Error: {str(e)[:200]}"
                    
                    st.session_state.chat_history.append(("assistant", response))
                    st.rerun()

# TAB 2: Chart
with tab2:
    st.markdown('<h1 style="text-align:center; font-size:2.5rem;">Live S&P 500 Chart</h1>', unsafe_allow_html=True)
    
    iframe = """
    <iframe src="https://s.tradingview.com/widgetembed/?symbol=FRED:SP500&interval=D&theme=light"
            width="100%" height="700" frameborder="0" allowfullscreen></iframe>
    """
    components.html(iframe, height=700)
    
    st.markdown("---")
    st.subheader("📰 Market News")
        
    try:
        news_data = finnhub_client.general_news('general',min_id=0)
        for article in news_data[:10]:
            st.markdown(f"**{article['headline']}**")
            date = datetime.fromtimestamp(article['datetime']).strftime('%b %d, %Y')
            st.caption(date)
            st.markdown(f"[Read more]({article['url']})")
            st.markdown("---")
    except Exception as e:
        st.error(f"Error: {e}")
        

# TAB 3: Dashboard
with tab3:
    if st.session_state.data_store['quant_df'] is not None:
        quant_df_display = st.session_state.data_store['quant_df'].copy()
        
        try:
            constituents_df = pd.read_csv('constituents.csv')
            quant_df_display = quant_df_display.merge(
                constituents_df[['Symbol', 'Security']], 
                left_on='Stock', 
                right_on='Symbol', 
                how='left'
            )
            quant_df_display.rename(columns={'Security': 'Company Name'}, inplace=True)
            quant_df_display['Company Name'].fillna(quant_df_display['Stock'], inplace=True)
        except:
            quant_df_display['Company Name'] = quant_df_display['Stock']
        
        st.subheader("📊 Historical Data Insights")
        
        insight_tab1, insight_tab2 = st.tabs(["🏆 Top Returns", "📉 Risk Analysis"])
        
        with insight_tab1:
            st.markdown("### Highest Cumulative Returns")
            
            top_returns = quant_df_display.nlargest(10, 'Cumulative Return (%)')
            top_returns['Display'] = top_returns['Stock'] + ' - ' + top_returns['Company Name']
            
            fig1 = px.bar(
                top_returns,
                x='Cumulative Return (%)',
                y='Display',
                orientation='h',
                title='Top 10 Stocks by Total Return',
                color='Cumulative Return (%)',
                color_continuous_scale='Greens',
                text='Cumulative Return (%)',
                hover_data={'Display': False, 'Stock': True, 'Company Name': True, 'Cumulative Return (%)': ':.2f'}
            )
            fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig1.update_layout(height=500, showlegend=False, yaxis_title="Company")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("### 📋 Top 15 Historical Performers")
            display_df = top_returns.head(15)[['Stock', 'Company Name', 'Cumulative Return (%)', 
                                               'Mean Daily Return (%)', 'Volatility (%)', 
                                               'Mean Close', 'Min Close', 'Max Close']].copy()
            display_df.columns = ['Ticker', 'Company', 'Total Return (%)', 'Avg Daily Return (%)', 
                                  'Volatility (%)', 'Avg Price ($)', 'Min Price ($)', 'Max Price ($)']
            display_df = display_df.round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with insight_tab2:
            st.markdown("### Volatility Insights")
            
            top_volatile = quant_df_display.nlargest(10, 'Volatility (%)')
            top_volatile['Display'] = top_volatile['Stock'] + ' - ' + top_volatile['Company Name']
            
            fig5 = px.bar(
                top_volatile,
                x='Volatility (%)',
                y='Display',
                orientation='h',
                title='Top 10 Most Volatile Stocks',
                color='Volatility (%)',
                color_continuous_scale='Reds',
                text='Volatility (%)',
                hover_data={'Display': False, 'Stock': True, 'Company Name': True, 'Volatility (%)': ':.2f'}
            )
            fig5.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig5.update_layout(height=500, showlegend=False, yaxis_title="Company")
            st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📈 Historical Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", len(quant_df))
    with col2:
        positive = len(quant_df[quant_df['Cumulative Return (%)'] > 0])
        st.metric("Positive Returns", f"{positive} ({(positive/len(quant_df)*100):.1f}%)")
    with col3:
        best = quant_df.loc[quant_df['Cumulative Return (%)'].idxmax()]
        st.metric("Best Performer", best['Stock'], f"{best['Cumulative Return (%)']:.2f}%")
    with col4:
        avg = quant_df['Cumulative Return (%)'].mean()
        st.metric("Average Return", f"{avg:.2f}%")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🤖 LLM Settings")
    llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini"])
    
    if llm_provider == "OpenAI":
        models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
    else:
        models = ["gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
    
    selected_model = st.selectbox("Model", models)
    api_key = st.text_input("API Key", type="password")
    
    if st.button("Initialize Agent") and api_key:
        with st.spinner("Initializing..."):
            agent = initialize_agent_custom(llm_provider, api_key, selected_model, news_df, quant_df)
            if agent:
                st.session_state.agent = agent
                st.session_state.llm_provider = llm_provider
                st.session_state.selected_model = selected_model
                st.session_state.api_key = api_key
                st.success("✅ Ready!")
                st.info("💡 Ask detailed questions for comprehensive answers with facts and analysis")
    
    st.divider()
    
    st.subheader("💰 Investment Parameters")
    investment_amount = st.number_input("Investment ($)", 1000, 1000000, 10000, 1000)
    num_stocks = st.slider("# Stocks", 3, 20, 5)
    risk_level = st.select_slider("Risk", ["Low", "Medium", "High"], value="Medium")
    
    if st.button("🎯 Generate Portfolio"):
        if st.session_state.agent:
            st.session_state.generate_portfolio = True
            st.session_state.portfolio_params = {
                'investment_amount': investment_amount,
                'num_stocks': num_stocks,
                'risk_level': risk_level
            }
            st.rerun()
        else:
            st.warning("Initialize agent first!")
    
    st.markdown("---")
    
    if st.button("🔄 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("⚠️ Educational use only")