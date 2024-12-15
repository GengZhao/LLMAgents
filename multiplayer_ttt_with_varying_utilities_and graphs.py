from typing import Dict, List
from autogen import ConversableAgent
from autogen import UserProxyAgent
import itertools
import sys
import os
import matplotlib.pyplot as plt

class Game():
    """
    A basic game class.

    Attributes:
        name (str): The name of the game.
        players (list): A list of players in the game.
        score (dict): A dictionary to store the score of each player.
    """

    def __init__(self):
        """
        Initializes a new game.

        Args:
            name (str): The name of the game.
        """
        self.players = [None] * 2
        self.actions = [None] * 2
        self.results = [None] * 2
        self.utilities = [None] * 2

    def game_interaction_prompt(self, history) -> str:
        return None

    def take_action(self, resp: str, player_id: int):
        return

    def end_turn(self):
        return

    def observation(self, player_id: int):
       return
    def game_summary(self, player_id: int):
        return self.observation(player_id)
    def utility(self, player_id: int):
        return

class RockPaperScissorsMultiplayerWithVaryingUtilities(Game):
    """
    A basic game class.

    Attributes:
        name (str): The name of the game.
        players (list): A list of players in the game.
        score (dict): A dictionary to store the score of each player.
    """

    def __init__(self, num_players):
        """
        Initializes a new game.

        Args:
            num_players (int): The number of players in the game.
        """
        self.num_players = num_players
        self.players = [None] * num_players
        self.actions = [None] * num_players
        self.results = [[None]*num_players] * num_players
        self.utilities = [[0] * num_players for _ in range(num_players)]
        self.utilitiesMaps = [{
            "rock": {'tie' : 0, 'win': 1, 'lose': 0},
            "paper": {'tie' : 0, 'win': 1, 'lose': 0},
            "scissors": {'tie' : 0, 'win': 1, 'lose': 0}, None: {'tie' : 0, 'win': 0, 'lose': 0}}] * num_players
        self.scores = [0] * num_players
        self.winning_conditions = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

        #modifies one player's utility
        self.utilitiesMaps[0] = {
            "rock": {'tie' : 0, 'win': 2, 'lose': 0},
            "paper": {'tie' : 0, 'win': 1, 'lose': 0},
            "scissors": {'tie' : 0, 'win': 1, 'lose': 0},
            None: {'tie' : 0, 'win': 0, 'lose': 0}
        }
    def take_action(self, resp: str, player_id: int):
        """Parses the action from the response string and updates the player's action.

        Args:
            resp: The response string.
            player_id: The ID of the player.
        """
        actions = ["rock", "paper", "scissors"]
        player_action = None  # Initialize player_action to None

        # Iterate through the words in reverse order to find the last occurrence
        for word in reversed(resp.split()):
            for a in actions:
                if a in word.lower():
                    player_action = a
                    break  # Break the inner loop once an action is found
            if player_action:
                break  # Break the outer loop if an action has been found

        if player_action:
            self.actions[player_id] = player_action

    def game_interaction_prompt(self, history, player_id) -> str:
        prompt = f"You are playing a game of rock, paper, scissors. This game is multiplayer, meaning you play against all people in the group at the same time. Aim to maximize utility. Choose 'rock', 'paper', or 'scissors'. You must choose, even if you don't have a history. \n"
        prompt += "Utilities:" + str(self.utilitiesMaps[player_id])
        prompt += f"\nLast 30 games results:\n"
        for i, round_info in enumerate(history[-30:]):  # Show last 5 rounds
            prompt += f"Round {i}: {round_info}\n"
        return prompt
    def end_turn(self):
        # TODO: double check: are the updates correct (e.g., any overwriting?)
        # Get the actions of all players
        print(self.actions)
        for i in range(len(self.players)):
            for j in range(len(self.players)):
                if i == j:
                    pass
                player1_action = self.actions[i]
                player2_action = self.actions[j] 
                print(player1_action)
                print(player2_action)
                # Determine the winner based on Rock-Paper-Scissors logic
                if player1_action == player2_action:
                    # It's a tie
                    #self.results[i][j] = "tie"
                    #self.results[j][i] = "tie"
                    self.utilities[i][j] = self.utilitiesMaps[i][player1_action]['tie']
                    #self.utilities[j][i] = self.utilitiesMaps[j][player2_action]['tie']
                elif self.winning_conditions[player1_action] == player2_action:
                    # Player 1 wins
                    #self.results[i][j] = "win"
                    #self.results[j][i] = "lose"
                    self.utilities[i][j] = self.utilitiesMaps[i][player1_action]['win']
                    #self.utilities[j][i] = self.utilitiesMaps[j][player2_action]['lose']
                else:
                    # Player 2 wins
                    #self.results[i][j] = "lose"
                    #self.results[j][i] = "win"
                    self.utilities[i][j] = self.utilitiesMaps[i][player1_action]['lose']
                    #self.utilities[j][i] = self.utilitiesMaps[j][player2_action]['win']
                print(self.utilities)

        # Update the score
        for i in range(self.num_players):
            self.scores[i] += sum(self.utilities[i])
        print(self.scores)

    def observation(self, player_id: int):
        #returns a string that says loses or wins, and also what the other player played and what you played. 
        observations = []
        for i in range(self.num_players):
            if i != player_id:
                other_player_action = self.actions[i]
                player_action = self.actions[player_id]
                utility = self.utilities[player_id][i]
                observations.append(f"You against player {i+1}, with utility {utility}. You played {player_action} and player {i+1} played {other_player_action}.")
        #print(self.results)
        return "\n".join(observations)

    def winner(self, player_id: int):
        return self.results[player_id] == "win"



   

class Experiment:
    def __init__(self, agents: List[ConversableAgent], game: Game):
        """
        Initializes a new Experiment.

        Args:
            name (str): The name of the game.
        """
        self.agents = agents
        self.game = game
        self.agent_states = {agent.name: {'cumulative_utility': 0, 'history': [], 'wins': 0, 'total_games': 0} for agent in agents}
        self.current_round = 0
        self.utility_history = {agent.name: [] for agent in agents} # Store utility after each round for each agent

    def execute_round(self) -> None:
        self.play_single_game(self.agents)
        self.current_round += 1

    def play_single_game(self, players: List[ConversableAgent]):
        gamemaker = ConversableAgent("Coordinator",
                                        system_message="",
                                        llm_config=False)

        #gets actions from agents and plays it in the game.
        for i, player in enumerate(players):
            player_history = self.agent_states[player.name]['history']
            result = gamemaker.initiate_chats(
                [
                    {
                    "recipient" : player,
                    "message" : self.game.game_interaction_prompt(player_history, i),
                    "max_turns" : 1,
                    "summary_method" : "last_msg"
                    }
                ]
                )
            action = result[0].summary
            self.game.take_action(action, i)
        
        self.game.end_turn()

        #updates state for players
        for i, player in enumerate(players):
            print(self.game.game_summary(i))
            self.agent_states[player.name]['history'] += [self.game.game_summary(i)]
            self.agent_states[player.name]['total_games'] += 1
            print(self.game.scores[i])
            self.agent_states[player.name]['cumulative_utility'] += self.game.scores[i]
            self.utility_history[player.name].append(self.game.scores[i]) # Store utility for plotting
            if self.game.winner(i):
                self.agent_states[player.name]['wins'] += 1
    
    def get_win_rates(self) -> Dict[str, float]:
        win_rates = {}
        for agent_name, state in self.agent_states.items():
            total_games = state['total_games']
            if total_games == 0:
                win_rate = 0.0
            else:
                win_rate = state['wins'] / total_games
            win_rates[agent_name] = win_rate
        return win_rates
    def get_scores(self) -> Dict[str, float]:
        scores = {}
        for player, state in self.agent_states.items():
            scores[player] = self.agent_states[player]['cumulative_utility'] 
        return scores

    def run_simulation(self, rounds: int, game_type: str = None) -> Dict[str, float]:
        if game_type == "two_player":
            for i in range(rounds):
                self.execute_round_two_player()
        else:
            for i in range(rounds):
                self.execute_round()
        
    
    def get_history_actions(self) -> Dict[str, List[str]]:
        history_actions = {}
        for agent_name, state in self.agent_states.items():
            history_actions[agent_name] = state['history']
        return history_actions
            
    def plot_utility_history(self):
        """Plots the utility history for each agent."""
        plt.figure(figsize=(10, 6))
        for agent_name, utilities in self.utility_history.items():
            plt.plot(utilities, label=agent_name)
        plt.xlabel("Round")
        plt.ylabel("Utility")
        plt.title("Utility Over Time for Each Agent")
        plt.legend()
        plt.grid(True)
        plt.show()






        







def main():
    # Read the API keys from the environment
    #config_list_gemini = {"cache_seed": None, "config_list": [{"model": "gemini-1.5-flash", "api_key": os.environ.get("GEMINI_API_KEY"), "api_type": "google"}]}
    llm_config = {"cache_seed": None, "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    llm_config_1 = {
        "cache_seed": None,
        "config_list": [
             {
                "model": "qwq",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
            }
        ]
    }
    #llm_config = {"cache_seed": None, "config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    player_prompt = "You are a rock paper scissors player. Each person has unique utilities. The person with the highest total utility wins."
    player_agent_1 = ConversableAgent("player_1",
                                        system_message=player_prompt,
                                        llm_config=llm_config)
    llm_config = {"cache_seed": None, "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    player_prompt = "You are a rock paper scissors player. Each person has unique utilities. The person with the highest total utility wins."
    player_agent_2 = ConversableAgent("player_2",
                                        system_message=player_prompt,
                                        llm_config=llm_config) # config_list_gemini)
    
    player_prompt = "You are a rock paper scissors player. Each person has unique utilities. The person with the highest total utility wins."
    player_agent_3 = ConversableAgent("player_3",
                                         system_message=player_prompt,
                                         llm_config=llm_config)
                                        
    game = RockPaperScissorsMultiplayerWithVaryingUtilities(3)
    experiment = Experiment([player_agent_1, player_agent_2, player_agent_3], game)
    experiment.run_simulation(100, "multi-player")
    # game = RockPaperScissorsMultiplayer(3)
    # experiment = Experiment([player_agent_1, player_agent_2, player_agent_3], game)
    # experiment.run_simulation(7, "multi-player")
    #game = RockPaperScissors()
    #experiment = Experiment([player_agent_1, player_agent_2], game)
    #experiment.run_simulation(5, "two_player")
    #print(experiment.agent_states) 
    print(experiment.get_scores())
    print(experiment.get_history_actions())
    experiment.plot_utility_history()

   
    

main()
