import autogen
import random
from typing import List, Tuple, Dict
import re
import matplotlib.pyplot as plt
import os

def extract_last_numeric_value(response: str) -> float:
    """
    Extract the last numeric value from the agent's response.

    Args:
        response (str): The response text from the agent.

    Returns:
        float: The last numeric value found in the response.

    Raises:
        ValueError: If no numeric value is found in the response.
    """
    # Use regex to find all numeric values (integer or float)
    matches = re.findall(r'-?\d+(?:\.\d+)?', response)
    if matches:
        # Take the last match as the numeric value
        return float(matches[-1])
    else:
        raise ValueError("No numeric value found in the response.")


class Agent:
    """Base class for an agent in the trading simulation."""
    
    def __init__(self, name: str, role: str, value: float, config_info: Dict, verbose: bool = False):
        """
        Initialize the agent with a name, role (buyer/seller), value, and specific configuration.
        """
        self.name = name
        self.role = role
        self.value = value
        self.config_info = config_info
        self.verbose = verbose
        self.agent = autogen.ConversableAgent(
            name=name,
            system_message=self.generate_prompt(),
            llm_config={'cache_seed': None, "config_list": [self.config_info]}
        )
    
    def generate_prompt(self) -> str:
        """
        Generate the system message prompt for the agent based on its role.
        """
        if self.role == "buyer":
            return (f"You are a buyer of space rockets on Mars. "
                    f"Your valuation (maximum willingness to pay) is {self.value}. "
                    f"Based on trading history and your valuation, submit a bid price. "
                    "Your goal is to maximize utility, which is (valuation - purchase price) if a transaction occurs and 0 otherwise. "
                    "Indicate your bid in the last line with ONLY a number.")
        elif self.role == "seller":
            return (f"You are a seller of space rockets on Mars. "
                    f"Your production cost (minimum acceptable price) is {self.value}. "
                    f"Based on trading history and your cost, submit an ask price. "
                    "Your goal is to maximize utility, which is (sale price - cost) if a transaction occurs and 0 otherwise. "
                    "Indicate your ask in the last line with ONLY a number.")
        else:
            raise ValueError("Role must be 'buyer' or 'seller'")
    
    def get_response(self, history: List[Tuple[float, bool]]) -> float:
        """
        Generate a response (bid/ask) from the agent based on trading history.
        """
        history_summary = f"History of {'bids' if self.role == 'buyer' else 'asks'}, trade/no trade:\n{history[-16:]}"
        response = self.agent.generate_reply(
            messages=[{"role": "user", "content": history_summary}]
        )
        if self.verbose:
            # Log prompt and response
            print(f"{self.role.capitalize()} {self.name} prompt: {history_summary}")
            print(f"{self.role.capitalize()} {self.name} response: {response}")
        try:
            return extract_last_numeric_value(response)
        except ValueError as e:
            print(f"Error parsing response from {self.name}: {response}")
            raise e


class Game:
    """Class for simulating the trading game."""
    
    def __init__(self, buyers_info: List[Tuple[str, float, Dict]], sellers_info: List[Tuple[str, float, Dict]], verbose: bool = False):
        """
        Initialize the game with buyer and seller information, including individual configurations.
        """
        self.buyers = [Agent(name, "buyer", valuation, config_info, verbose=verbose) for name, valuation, config_info in buyers_info]
        self.sellers = [Agent(name, "seller", cost, config_info, verbose=verbose) for name, cost, config_info in sellers_info]
        self.trading_history = []
    
    def run_round(self, public_signals=False) -> None:
        """
        Run a single round of the trading game.
        """
        # Shuffle buyers and sellers for random pairing
        random.shuffle(self.buyers)
        random.shuffle(self.sellers)
        
        new_records = []
        # Pair buyers and sellers
        for buyer, seller in zip(self.buyers, self.sellers):
            # Get responses
            if public_signals: # Sees all bids and asks
                bid = buyer.get_response([(bid, trade) for _, _, bid, _, trade in self.trading_history])
                ask = seller.get_response([(ask, trade) for _, _, _, ask, trade in self.trading_history])
            else: # Only sees own bid and ask
                bid = buyer.get_response([(bid, trade) for b, s, bid, _, trade in self.trading_history if b == buyer.name])
                ask = seller.get_response([(ask, trade) for b, s, _, ask, trade in self.trading_history if s == seller.name])
            
            # Determine trade
            trade_executed = bid >= ask
            new_records.append((buyer.name, seller.name, bid, ask, trade_executed))
            
            # Print round details
            print(f"Buyer: {buyer.name} Bid: ${bid:.2f}")
            print(f"Seller: {seller.name} Ask: ${ask:.2f}")
            print(f"Trade executed: {trade_executed}\n")
        self.trading_history.extend(new_records)
    
    def run_simulation(self, num_rounds: int, public_signals: bool = False) -> List[Tuple[str, str, float, float, bool]]:
        """
        Run the simulation for a specified number of rounds.
        """
        for round in range(num_rounds):
            print(f"\nRound {round + 1}")
            self.run_round(public_signals=public_signals)
        return self.trading_history


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Define buyer and seller information with individual configurations
buyers_info = [
    ("buyer1", 115.0, {"model": "gpt-4o-mini", "temperature": 1.0, "api_key": OPENAI_API_KEY}),
    ("buyer2", 101.0, {"model": "gpt-4o-mini", "temperature": 1.0, "api_key": OPENAI_API_KEY}),
    ("buyer3", 97.0, {"model": "gpt-4o-mini", "temperature": 1.0, "api_key": OPENAI_API_KEY}),
]

sellers_info = [
    ("seller1", 89.0, {"model": "gpt-4o-mini", "temperature": 1.0, "api_key": OPENAI_API_KEY}),
    ("seller2", 99.0, {"model": "gpt-4o-mini", "temperature": 1.0, "api_key": OPENAI_API_KEY}),
    ("seller3", 96.0, {"model": "gpt-4o-mini", "temperature": 1.0, "api_key": OPENAI_API_KEY}),
]

# Run the simulation
game = Game(buyers_info, sellers_info, verbose=True)
trading_history = game.run_simulation(num_rounds=10, public_signals=True)

# Print trading history
for trade in trading_history:
    buyer, seller, bid, ask, executed = trade
    print(f"Buyer: {buyer}, Seller: {seller}, Bid: ${bid:.2f}, Ask: ${ask:.2f}, Trade Executed: {executed}")

# Based on the trade history, make a plot showing bid and ask prices over time for each buyer and seller.
# Initialize figure
plt.figure(figsize=(12, 6))

# For each buyer/seller in the game, extract bid/ask prices over time
for i, buyer in enumerate(game.buyers):
    bids = [bid for b, s, bid, ask, t in trading_history if b == buyer.name]
    rounds = range(1, len(bids) + 1)
    # Plot buyer bids
    plt.plot(rounds, bids, label=f"{buyer.name} Bids", marker="o", color=f"C{i}")
    # Plot buyer valuation line. Use the same color as the bids.
    plt.axhline(y=buyer.value, color=f"C{i}", linestyle=":", label=f"{buyer.name} Valuation")

for i, seller in enumerate(game.sellers):
    asks = [ask for b, s, bid, ask, t in trading_history if s == seller.name]
    rounds = range(1, len(asks) + 1)
    # Plot seller asks
    plt.plot(rounds, asks, label=f"{seller.name} Asks", marker="+", color=f"C{i}", linestyle="--")
    # Plot seller cost line. Use the same color as the asks.
    plt.axhline(y=seller.value, color=f"C{i}", linestyle="-.", label=f"{seller.name} Cost")


# Add labels and legend
plt.xlabel("Round")
plt.ylabel("Price (Million Martian Dollars)")
plt.title("Bid and Ask Prices Over Time")
plt.legend()
plt.grid(True)
plt.show()