from openai import OpenAI
from crewai import Agent, Task, LLM
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import logging
from datetime import datetime

from Interaction_crew_ai import InteractionAgentSystem, InteractionState

# class OpenAIAgent(Agent):
#     """Custom agent class using OpenAI API directly"""
#     def __init__(self, role: str, model_name: str = "gpt-4o-mini"):
#         self.model_name = model_name
#         super().__init__(
#             role=role,
#             goal="Win at Rock-Paper-Scissors by analyzing patterns and adapting strategy",
#             backstory="You are an intelligent agent playing a Rock-Paper-Scissors tournament. "
#                      "You can analyze past performance to inform your strategy.",
#             allow_delegation=False,
#             verbose=True
#         )

#     async def execute_task(self, task: str) -> str:
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "system", "content": self.backstory},
#                     {"role": "user", "content": task}
#                 ],
#                 temperature=0.7,
#                 max_tokens=150
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             logging.error(f"Error executing task: {e}")
#             return "Rock"  # Default action in case of API error

class RPSExperiment(InteractionAgentSystem):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        # Set up OpenAI API key
        # os.environ["OPENAI_API_KEY"] = api_key
        
        # Create three agents with specified model
        agents = [
            Agent(
                role=f"Player_{i}",
                goal="Win at Rock-Paper-Scissors by analyzing patterns and adapting strategy",
                backstory="You are an intelligent agent playing a Rock-Paper-Scissors tournament. "
                         "You can analyze past performance to inform your strategy.",
                allow_delegation=False,
                verbose=True,
                llm=LLM(model=model_name, max_tokens=80)
            )
            for i in range(1, 4)
        ]

        # Game-specific parameters
        system_message = """
        You are participating in a Rock-Paper-Scissors tournament against two other players.
        """
        # Each round, you will receive your win and loss rates against each opponent from the previous round.
        # Your goal is to maximize your winning percentage by choosing the optimal action each round.

        action_set = ["Rock", "Paper", "Scissors"]
        observation_set = [f"{result} against Player_{i}"  
                         for result in ["win", "loss", "tie"] 
                         for i in range(1, 4)]

        # Initialize the parent class
        super().__init__(
            agents=agents,
            system_message=system_message,
            action_set=action_set,
            observation_set=observation_set,
            interaction_function=self._rps_interaction,
            max_rounds=3,
        )

        # Initialize RPS-specific state tracking
        self.state.global_state.update({
            'win_counts': defaultdict(lambda: defaultdict(int)),
            'action_counts': defaultdict(lambda: defaultdict(int)),
            'total_games': defaultdict(int),
            'start_time': datetime.now(),
            'api_calls': 0,
            'errors': 0
        })

    def _determine_winner(self, action1: str, action2: str) -> int:
        """Returns 1 if action1 wins, -1 if action2 wins, 0 if tie"""
        if action1 == action2:
            return 0
        winning_moves = {
            "Rock": "Scissors",
            "Paper": "Rock",
            "Scissors": "Paper"
        }
        return 1 if winning_moves[action1] == action2 else -1

    def _calculate_win_rates(self, player: str) -> Dict[str, Dict[str, float]]:
        """Calculate win/loss rates against each opponent"""
        rates = {}
        for opponent in self.agents.keys():
            if opponent != player:
                total_games = self.state.global_state['total_games'][f"{player}_vs_{opponent}"]
                if total_games > 0:
                    win_rate = (self.state.global_state['win_counts'][player][opponent] / total_games)
                    loss_rate = (self.state.global_state['win_counts'][opponent][player] / total_games)
                else:
                    win_rate = loss_rate = float('nan')
                rates[opponent] = {"win_rate": win_rate, "loss_rate": loss_rate}
        return rates

    def _rps_interaction(self, actions: Dict[str, str], 
                        state: InteractionState) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Process one round of RPS interactions"""
        # Update action counts and API stats
        state.global_state['api_calls'] += len(actions)
        
        for player, action in actions.items():
            state.global_state['action_counts'][player][action] += 1

        # Process all pairwise interactions
        new_observations = {}
        for player1 in self.agents.keys():
            # Create observation string for each player
            results = self._calculate_win_rates(player1)
            observation_parts = []
            for opponent, rates in results.items():
                observation_parts.append(
                    f"Against {opponent}: "
                    f"Win rate = {rates['win_rate']:.2%}, "
                    f"Loss rate = {rates['loss_rate']:.2%}"
                )
            new_observations[player1] = "\n".join(observation_parts)

            # Process games against each opponent
            for player2 in self.agents.keys():
                if player1 < player2:  # Process each pair once
                    result = self._determine_winner(actions[player1], actions[player2])
                    match_id = f"{player1}_vs_{player2}"
                    state.global_state['total_games'][match_id] += 1
                    
                    if result == 1:
                        state.global_state['win_counts'][player1][player2] += 1
                    elif result == -1:
                        state.global_state['win_counts'][player2][player1] += 1

        return new_observations, {'global': state.global_state}

    def get_experiment_results(self) -> Dict[str, Any]:
        """Generate comprehensive results of the experiment"""
        results = {
            'win_rates': {},
            'action_distributions': {},
            'overall_performance': {},
            'experiment_stats': {
                'duration': datetime.now() - self.state.global_state['start_time'],
                'total_api_calls': self.state.global_state['api_calls'],
                'error_count': self.state.global_state['errors'],
                'total_rounds': self.state.round
            }
        }

        # Calculate final win rates for each player
        for player in self.agents.keys():
            results['win_rates'][player] = self._calculate_win_rates(player)

        # Calculate action distributions
        total_rounds = self.state.round
        for player in self.agents.keys():
            action_counts = self.state.global_state['action_counts'][player]
            distribution = {
                action: count / total_rounds 
                for action, count in action_counts.items()
            }
            results['action_distributions'][player] = distribution

        # Calculate overall performance
        for player in self.agents.keys():
            total_wins = sum(self.state.global_state['win_counts'][player].values())
            total_games = sum(self.state.global_state['total_games'][f"{player}_vs_{opponent}"]
                            for opponent in self.agents.keys() if opponent != player)
            results['overall_performance'][player] = {
                'total_wins': total_wins,
                'total_games': total_games,
                'win_percentage': total_wins / total_games if total_games > 0 else 0
            }

        return results

# Example usage:
async def run_rps_experiment(api_key: str, model_name: str = "gpt-4o-mini"):
    experiment = RPSExperiment(api_key=api_key, model_name=model_name)
    await experiment.run_interaction()
    
    results = experiment.get_experiment_results()
    
    # Print comprehensive results
    print("\nExperiment Results:")
    print("\nExperiment Statistics:")
    stats = results['experiment_stats']
    print(f"Duration: {stats['duration']}")
    print(f"Total API Calls: {stats['total_api_calls']}")
    print(f"Errors: {stats['error_count']}")
    print(f"Total Rounds: {stats['total_rounds']}")
    
    print("\nAction Distributions:")
    for player, dist in results['action_distributions'].items():
        print(f"\n{player}:")
        for action, prob in dist.items():
            print(f"{action}: {prob:.2%}")
    
    print("\nOverall Performance:")
    for player, stats in results['overall_performance'].items():
        print(f"\n{player}:")
        print(f"Win Rate: {stats['win_percentage']:.2%}")
        print(f"Total Wins: {stats['total_wins']}")
        print(f"Total Games: {stats['total_games']}")
    
    return results

if __name__ == "__main__":
    import asyncio
    
    os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
    # Replace with your OpenAI API key
    API_KEY = os.environ.get("OPENAI_API_KEY")

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the experiment
    results = asyncio.run(run_rps_experiment(
        api_key=API_KEY,
        model_name="gpt-4o-mini",
    ))