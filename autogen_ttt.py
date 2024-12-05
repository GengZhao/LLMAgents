from typing import Dict, List
from autogen import ConversableAgent
from autogen import UserProxyAgent
import itertools
import sys
import os

class Game:
    def __init__(self):
        self.players = [None] * 2
        self.actions = [None] * 2
        self.results = [None] * 2
        self.utilities = [None] * 2

    def game_interaction_prompt(self, history):
        raise NotImplementedError("This method should be implemented in a subclass")

    def take_action(self, resp: str, player_id: int):
        raise NotImplementedError("This method should be implemented in a subclass")

    def end_turn(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def observation(self, player_id: int):
        raise NotImplementedError("This method should be implemented in a subclass")

    def game_summary(self, player_id: int):
        raise NotImplementedError("This method should be implemented in a subclass")

    def utility(self, player_id: int):
        raise NotImplementedError("This method should be implemented in a subclass")


        

class TicTacToe(Game):

    def __init__(self):
        super().__init__()
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = 0
        self.winner = None

    def game_interaction_prompt(self, history) -> str:
        # Generate the updated board state
        board_state = "\n".join([" | ".join(row) for row in self.board])
        
        # Build the prompt with the updated board and game history
        prompt = f"You are playing Tic-Tac-Toe.\n"
        prompt += f"Board state:\n{board_state}\n"
        prompt += f"Choose your move as 'row,column' (e.g., '1,2' for the top-middle cell).\n"
        
        if history:
            prompt += "Game history (last 5 moves):\n"
            prompt += "\n".join(history[-5:])
            prompt += "\n"

        return prompt

    def take_action(self, resp: str, player_id: int):
        try:
            # Extract the numeric coordinates from the response
            import re
            matches = re.findall(r"\d,\d", resp)  # Find all occurrences of 'digit,digit'
            if not matches:
                raise ValueError("No valid move found in the response.")

            # Use the first valid match as the move
            move = matches[0]  # Expecting at least one valid 'row,column' pair
            row, col = map(int, move.split(","))  # Split into integers

            # Convert 1-based player input to 0-based Python indexing
            row -= 1
            col -= 1

            # Validate the move and update the board
            if 0 <= row < 3 and 0 <= col < 3:
                if self.board[row][col] == " ":
                    # Mark the board with the player's symbol
                    self.board[row][col] = "X" if player_id == 0 else "O"
                    print(f"Player {player_id + 1} placed {'X' if player_id == 0 else 'O'} at ({row + 1}, {col + 1}).")
                else:
                    raise ValueError(f"Cell ({row + 1}, {col + 1}) is already occupied.")
            else:
                raise ValueError(f"Move ({row + 1}, {col + 1}) is out of bounds.")
        except Exception as e:
            print(f"Invalid move format from player {player_id}: {resp}")
            raise e



    def check_winner(self):
        # Check rows and columns for a win
        for i in range(3):
            # Check row i
            if self.board[i][0] != " " and all(self.board[i][j] == self.board[i][0] for j in range(3)):
                self.winner = 0 if self.board[i][0] == "X" else 1
                return

            # Check column i
            if self.board[0][i] != " " and all(self.board[j][i] == self.board[0][i] for j in range(3)):
                self.winner = 0 if self.board[0][i] == "X" else 1
                return

        # Check diagonals for a win
        if self.board[0][0] != " " and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            self.winner = 0 if self.board[0][0] == "X" else 1
            return

        if self.board[0][2] != " " and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            self.winner = 0 if self.board[0][2] == "X" else 1
            return

        # Check for a tie (board full and no winner)
        if all(cell != " " for row in self.board for cell in row):
            self.winner = "tie"


    def end_turn(self):
        self.check_winner()
        if self.winner is not None:
            if self.winner == "tie":
                self.results = ["tie", "tie"]
                self.utilities = [0, 0]
            else:
                self.results = ["win", "lose"] if self.winner == 0 else ["lose", "win"]
                self.utilities = [1, 0] if self.winner == 0 else [0, 1]
        else:
            self.results = ["ongoing", "ongoing"]
            self.utilities = [0, 0]



    def observation(self, player_id: int):
        board_state = "\n".join([" | ".join(row) for row in self.board])
        return f"Board state:\n{board_state}\n"


    def game_summary(self, player_id: int):
        return f"Game result: {self.results[player_id]}. Final board state:\n" + \
               "\n".join([" | ".join(row) for row in self.board])

    
    def utility(self, player_id: int):
        return self.utilities[player_id]



class Experiment:
    """
    Handles the execution and management of experiments for games with multiple agents.

    Attributes:
        agents (list): The list of agents participating in the experiment.
        game (Game): The game being played.
        agent_states (dict): A dictionary to track the state of each agent.
        current_round (int): The current round number.
    """

    def __init__(self, agents: List[ConversableAgent], game: Game):
        
        self.agents = agents
        self.game = game
        self.agent_states = {agent.name: {'cumulative_utility': 0, 'history': [], 'wins': 0, 'total_games': 0} for agent in agents}
        self.current_round = 0

    def execute_round(self) -> None:
        self.play_single_game(self.agents)
        self.current_round += 1



    def play_single_game(self, players: List[ConversableAgent]):
        gamemaker = ConversableAgent(
            "Coordinator",
            system_message="",
            llm_config={"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
        )

        current_player_index = 0

        while self.game.winner is None:  
            current_player = players[current_player_index]
            player_history = self.agent_states[current_player.name]['history']
            prompt = self.game.game_interaction_prompt(player_history)

            result = gamemaker.initiate_chats(
                [
                    {
                        "recipient": current_player,
                        "message": prompt,
                        "max_turns": 1,
                        "summary_method": "last_msg"
                    }
                ]
            )

            action = result[0].summary
            try:
                self.game.take_action(action, current_player_index)
                self.game.end_turn()  

            
                if self.game.winner is not None:
                    if self.game.winner == "tie":
                        print("Game Over! It's a Draw!")
                    else:
                        print(f"Game Over! {'Player 1 (X)' if self.game.winner == 0 else 'Player 2 (O)'} wins!")
                    break  

                for i, player in enumerate(players):
                    self.agent_states[player.name]['history'].append(self.game.game_summary(i))
                    self.agent_states[player.name]['total_games'] += 1
                    self.agent_states[player.name]['cumulative_utility'] += self.game.utility(i)
                    if self.game.results[i] == "win":
                        self.agent_states[player.name]['wins'] += 1

                
                current_player_index = 1 - current_player_index
            except ValueError as e:
                
                print(f"Error: {e}")
                continue 






    def run_simulation(self, rounds: int) -> None:
        """
        Runs the simulation for the specified number of rounds.

        Args:
            rounds (int): Number of game rounds to simulate.
        """
        for _ in range(rounds):
            self.execute_round()

    def get_win_rates(self) -> Dict[str, float]:
        """
        Calculates and returns win rates for all agents.

        Returns:
            dict: A dictionary with agent names as keys and win rates as values.
        """
        win_rates = {}
        for agent_name, state in self.agent_states.items():
            total_games = state['total_games']
            win_rates[agent_name] = state['wins'] / total_games if total_games > 0 else 0.0
        return win_rates

    def get_history_actions(self) -> Dict[str, List[str]]:
        """
        Returns the action history for all agents.

        Returns:
            dict: A dictionary with agent names as keys and action history as values.
        """
        history_actions = {}
        for agent_name, state in self.agent_states.items():
            history_actions[agent_name] = state['history']
        return history_actions

class ExperimentTicTacToe(Experiment):
    """
    Custom experiment class for Tic-Tac-Toe to ensure two-player gameplay.
    """
    def execute_round(self) -> None:
        """
        Executes a single round specifically for Tic-Tac-Toe.
        """
        self.play_single_game(self.agents)

def main():
    llm_config = {"cache_seed": None, "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    player_1_prompt = "You are playing Tic-Tac-Toe as 'X'. Choose your move based on the board state provided, you may not choose postion that is already chosen by the other player."
    player_2_prompt = "You are playing Tic-Tac-Toe as 'O'. Choose your move based on the board state provided, you may not choose postion that is already chosen by the other player."

    player_agent_1 = ConversableAgent("Player 1", system_message=player_1_prompt, llm_config=llm_config)
    player_agent_2 = ConversableAgent("Player 2", system_message=player_2_prompt, llm_config=llm_config)

    game = TicTacToe()
    experiment = ExperimentTicTacToe([player_agent_1, player_agent_2], game)
    experiment.run_simulation(10)
    
    print(experiment.get_win_rates())
    print(experiment.get_history_actions())

if __name__ == "__main__":
    main()