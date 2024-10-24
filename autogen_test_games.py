from typing import Dict, List
from autogen import ConversableAgent
from autogen import UserProxyAgent
import sys
import os



class Game:
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

    def game_interaction_prompt(self, history: []) -> str:
            prompt = f"You are playing a game of rock, paper, scissors. Choose 'rock', 'paper', or 'scissors'.\n"
            prompt += f"Last 5 games results:\n"
            for i, round_info in enumerate(history[-5:]):  # Show last 5 rounds
                prompt += f"Round {i}: {round_info}\n"
            return prompt

    def take_action(self, action: str, player_id: int):
        #parses action
        actions = ["rock", "paper", "scissors"]
        for a in actions:
            if a in action.lower():
                #adds action
                self.actions[player_id] = a

    def end_turn(self):
        assert len(self.players) == 2, "there should be two players"
        # Get the actions of both players
        player1_action = self.actions[0]
        player2_action = self.actions[1]

        # Determine the winner based on Rock-Paper-Scissors logic
        if player1_action == player2_action:
            # It's a tie
            self.results = ["tie", "tie"]
            self.utilities = [0, 0]

        elif (player1_action == "rock" and player2_action == "scissors") or \
            (player1_action == "scissors" and player2_action == "paper") or \
            (player1_action == "paper" and player2_action == "rock"):
            # Player 1 wins
            self.results = ["win", "lose"]
            self.utilities = [1, 0]
        else:
            # Player 2 wins
            self.results = ["lose", "win"]
            self.utilities = [0, 1]

    def observation(self, player_id: int):
        #returns a string that says loses or wins, and also what the other player played and what you played. 
        outcome = self.results[player_id]

        other_player_id = 1 - player_id
        other_player_action = self.actions[other_player_id]
        player_action = self.actions[player_id]
        utility = self.utilities[player_id]

        return f"You {outcome}, with utility {utility}. You played {player_action} and your opponent played {other_player_action}."
    
    def game_summary(self, player_id: int):
        return self.observation(player_id)

    def utility(self, player_id: int):
        return self.utilities[player_id]

   

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


    def execute_round(self) -> None:
        for player_1 in self.agents:
            for player_2 in self.agents:
                if player_1.name != player_2.name:
                    self.play_single_game([player_1, player_2])
        self.current_round += 1


    def play_single_game(self, players: List[ConversableAgent]):
        gamemaker = ConversableAgent("Coordinator",
                                        system_message="",
                                        llm_config= {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]})

        #gets actions from agents and plays it in the game.
        for i, player in enumerate(players):
            player_history = self.agent_states[player.name]['history']
            result = gamemaker.initiate_chats(
                [
                    {
                    "recipient" : player,
                    "message" : self.game.game_interaction_prompt(player_history),
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
            self.agent_states[player.name]['history'] += [self.game.game_summary(i)]
            self.agent_states[player.name]['total_games'] += 1
            self.agent_states[player.name]['cumulative_utility'] += self.game.utility(i)
            if self.game.utility(i) == 1:
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
    def run_simulation(self, rounds: int) -> Dict[str, float]:
        for i in range(rounds):
            self.execute_round()

            






        







# Do not modify the signature of the "main" function.
def main():
    #entrypoint_agent_system_message = "You are working with another agent to help answer the user's query. The other agent is able to take in a list of reviews, and transform it into a list of rankings with a food score, and a customer service score. Please use this score in order to find the overall score. " # TODO
    # example LLM config for the entrypoint agent
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    # feed the resturaunt data into score extract agent
    player_prompt = "You are an aggressive rock paper scissors player."
    player_agent_1 = ConversableAgent("player_1",
                                        system_message=player_prompt,
                                        llm_config=llm_config)

    player_prompt = "You are an conservative rock paper scissors player."
    player_agent_2 = ConversableAgent("player_2",
                                        system_message=player_prompt,
                                        llm_config=llm_config)

    game = Game()
    experiment = Experiment([player_agent_1, player_agent_2], game)
    experiment.run_simulation(5)
    #print(experiment.agent_states) 
    print(experiment.get_win_rates())

   
    

main()

