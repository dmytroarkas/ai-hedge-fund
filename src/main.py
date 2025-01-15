from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init

from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.news_sentiment import news_sentiment_agent  # Импорт нового агента
from agents.reflexivity import reflexivity_agent  # Импорт нового агента
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)

def parse_hedge_fund_response(response):
    import json
    try:
        return json.loads(response)
    except:
        print(f"Error parsing response: {response}")
        return None

##### Run the Hedge Fund #####
def run_hedge_fund(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list = None,
):
    # Create a new workflow if analysts are customized
    if selected_analysts is not None:
        workflow = create_workflow(selected_analysts)
        agent = workflow.compile()
    else:
        agent = app

    final_state = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            },
        },
    )
    return {
        "decision": parse_hedge_fund_response(final_state["messages"][-1].content),
        "analyst_signals": final_state["data"]["analyst_signals"],
    }


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = [
            "technical_analyst", 
            "fundamentals_analyst", 
            "sentiment_analyst", 
            "valuation_analyst",
            "news_sentiment_agent",  # Добавлен новый агент по умолчанию
            "reflexivity_agent",  # Новый агент
        ]

    # Dictionary of all available analysts
    analyst_nodes = {
        "technical_analyst": ("technical_analyst_agent", technical_analyst_agent),
        "fundamentals_analyst": ("fundamentals_agent", fundamentals_agent),
        "sentiment_analyst": ("sentiment_agent", sentiment_agent),
        "valuation_analyst": ("valuation_agent", valuation_agent),
        "news_sentiment_agent": ("news_sentiment_agent", news_sentiment_agent),  # Новый агент
        "reflexivity_agent": ("reflexivity_agent", reflexivity_agent),  # Новый агент
    }

    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


def select_analysts():
    """Function for selecting analysts via text input."""
    print("Select analysts (enter numbers separated by commas):")
    print("1. Technical Analyst")
    print("2. Fundamentals Analyst")
    print("3. Sentiment Analyst")
    print("4. Valuation Analyst")
    print("5. News Sentiment Analyst")  # Новый пункт
    print("6. Reflexivity Analyst")  # Новый пункт
    choices = input("Input: ").strip().split(",")
    choices = [int(choice.strip()) for choice in choices if choice.strip().isdigit()]
    
    # Convert numbers to analyst keys
    analyst_keys = [
        "technical_analyst", 
        "fundamentals_analyst", 
        "sentiment_analyst", 
        "valuation_analyst",
        "news_sentiment_agent",  # Новый ключ
        "reflexivity_agent",  # Новый ключ
    ]
    selected_analysts = [analyst_keys[choice - 1] for choice in choices if 1 <= choice <= 6]
    
    if not selected_analysts:
        print("No analysts selected. Using all analysts by default.")
        return None
    else:
        print(f"\nSelected analysts: {', '.join(selected_analysts)}")
        return selected_analysts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument(
        "--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today"
    )
    parser.add_argument(
        "--show-reasoning", action="store_true", help="Show reasoning from each agent"
    )

    args = parser.parse_args()

    # Select analysts
    selected_analysts = select_analysts()

    # Create the workflow
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # TODO: Make this configurable via args
    portfolio = {
        "cash": 100000.0,  # $100,000 initial cash
        "stock": 0,  # No initial stock position
    }

    # Run the hedge fund
    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
    )
    print_trading_output(result)
