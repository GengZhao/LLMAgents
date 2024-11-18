from typing import Dict, List
from autogen import ConversableAgent
from autogen import UserProxyAgent
import re
import itertools
import sys
import os

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
        return
    def take_action(self, resp: str, player_id: int):
        return

    def end_turn(self):
        assert len(self.players) == 2, "there should be two players"

    def observation(self, player_id: int):
        return
    
    def game_summary(self, player_id: int):
        return

    def utility(self, player_id: int):
        return
    

class DivideDollar(Game):
    """
    a game for multiplayer dividing 100 cents
    """
    def __init__(self,num_players,num_rounds):
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.players = [None] * num_players
        self.actions = [None] * num_players
        self.results = [None] * num_players
        self.utilities = [None] * num_players
        self.score = {i: 0 for i in range(num_players)} 

    def take_action(self, resp:str, player_id:int):
        all_floats = re.findall(r"\d+\.\d+", resp)
        assert len(all_floats) >0, "no value found in response"
        final = float(all_floats[-1])
        self.actions[player_id] = final
    
    def game_interaction_prompt(self, history,round_id) -> str:
        prompt = f"This is round {round_id}, pick your bid value. \n"
        prompt += "history: "+ str(history)
        return prompt

    def end_turn(self):
        total = float(sum(self.actions))
        if total <= 100.0:
            self.results = self.actions
        else:
            self.results = [0.0]*self.num_players
        
        for i in range(self.num_players):
            self.score[i] += self.results[i]
        
    
    def observation(self, player_id: int):
        ob = f"your bid was {self.actions[player_id]} and the result from this round for all player was {self.results}"
        return ob



class Experiment:
    def __init__(self, agents: List[ConversableAgent], game: Game):
        """
        Initializes a new Experiment. for single game of DIVIDE DOLLAR

        Args:
            name (str): The name of the game.
        """
        self.agents = agents
        self.game = game
        self.agent_states = {agent.name: {'cumulative_score': 0, 'history': [], 'wins': 0, 'total_round': 0} for agent in agents}
        self.current_round = 0

    def execute_round(self) -> None:
        self.play_single_round(self.agents)
        self.current_round += 1

    def play_single_round(self, players: List[ConversableAgent]):
        gamemaker = ConversableAgent("Coordinator",
                                        system_message="",
                                        llm_config= {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]})

        #gets actions from agents and plays it in the game.
        for i, player in enumerate(players):
            player_history = self.agent_states[player.name]['history']
            message = self.game.game_interaction_prompt(player_history,self.current_round)
            result = gamemaker.initiate_chats(
                [
                    {
                    "recipient" : player,
                    "message" : message,
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
            self.agent_states[player.name]['history'].append(self.game.observation(i))
            self.agent_states[player.name]['total_round'] += 1
            self.agent_states[player.name]['cumulative_score'] = self.game.score[i]

    def run_simulation(self, rounds: int):
        for i in range(rounds):
            self.execute_round()

    def report(self):
        d = {}
        for i, player in enumerate(self.agents):
            d[i] = self.agent_states[player.name]['cumulative_score']
        return str(d)
    
def main():
    # Read the API keys from the environment
    config_list_gemini = {"cache_seed": None, "config_list": [{"model": "gemini-1.5-flash", "api_key": os.environ.get("GEMINI_API_KEY"), "api_type": "google"}]}
    llm_config = {"cache_seed": None, "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}


    num_player = 3
    num_round = 10
    rule = f"you are playing the game of Divide Dollar of {num_player} players.\n"
    rule += f"""In the Divide Dollar game, each player bids a number with exactly one decimal place (e.g., 12.3) in every round, the last line should be your bid value. 
    If the total sum of all players' bids in each round is less than 100.0, each player receives an amount equal to their individual bid.
        However, if the total sum is 100.0 or more, no one receives anything, and all players get 0.0 for that round. 
        This game consist of {num_round} rounds.\n
    """
    rule += "Whoever has the highest cumulative score wins."

    player_prompt = rule + "you are a random player, You should pick random value between 0.0 and 100.0"
    player_agent_1 = ConversableAgent("player_1",
                                        system_message=player_prompt,
                                        llm_config=llm_config)

    player_prompt = rule + "you are optimal player. You should always play optimally"
    player_agent_2 = ConversableAgent("player_2",
                                        system_message=player_prompt,
                                        llm_config=llm_config) # config_list_gemini)
    
    player_prompt = rule + "You are a smart player. Observe how how other plays and and make the best bid to maximize your cumulative score"
    player_agent_3 = ConversableAgent("player_3",
                                        system_message=player_prompt,
                                        llm_config=llm_config)

    game = DivideDollar(num_players=num_player,num_rounds=num_round)
    experiment = Experiment([player_agent_1, player_agent_2,player_agent_3], game)
    experiment.run_simulation(num_round)
    print(experiment.agent_states)
    print("--------\n")
    print(experiment.report())
   
    

main()