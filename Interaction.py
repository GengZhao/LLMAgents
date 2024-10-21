from typing import Dict, List, Callable, Tuple
from autogen import ConversableAgent
import random

class InteractionAgentSystem:
    def __init__(self, 
                 agents: List[ConversableAgent],
                 system_message: str,
                 action_set: List[str],
                 observation_set: List[str],
                 interaction_prompt: Callable[[List[str], List[str], List[str]], str],
                 result_parser: Callable[[str], Dict[str, str]],
                 utility_calculator: Callable[[Dict[str, str]], Dict[str, float]]) -> None:
        self.agents = agents
        self.system_message = system_message
        self.action_set = action_set
        self.observation_set = observation_set
        self.interaction_prompt = interaction_prompt
        self.result_parser = result_parser
        self.utility_calculator = utility_calculator
        self.agent_states = {agent.name: {'cumulative_utility': 0, 'history': []} for agent in agents}
        self.current_round = 0

        # Set up agents with system message
        for agent in self.agents:
            agent.reset()
            agent.send(self.system_message, silent=True)

    def run_simulation(self, num_rounds: int) -> Dict[str, float]:
        for _ in range(num_rounds):
            self.current_round += 1
            self._execute_round()
        
        return self._get_final_utilities()

    def _execute_round(self) -> None:
        # Generate the interaction prompt
        prompt = self.interaction_prompt(self.agents, self.action_set, self.observation_set)
        
        # Let the agents interact
        actions = {}
        for agent in self.agents:
            response = agent.generate_response(prompt)
            actions[agent.name] = response
        
        # Parse the result
        parsed_actions = self.result_parser(actions)
        
        # Calculate utilities
        utilities = self.utility_calculator(parsed_actions)
        
        # Update agent states
        for agent in self.agents:
            self._update_agent_state(agent, parsed_actions[agent.name], utilities[agent.name])

    def _update_agent_state(self, agent: ConversableAgent, action: str, utility: float) -> None:
        self.agent_states[agent.name]['history'].append({
            'round': self.current_round,
            'action': action,
            'utility': utility
        })
        self.agent_states[agent.name]['cumulative_utility'] += utility

    def _get_final_utilities(self) -> Dict[str, float]:
        return {agent.name: state['cumulative_utility'] for agent, state in zip(self.agents, self.agent_states.values())}

    def get_agent_history(self, agent_name: str) -> List[Dict]:
        return self.agent_states[agent_name]['history']

    def get_round_summary(self, round_num: int) -> Dict[str, Dict]:
        return {agent: [round_data for round_data in state['history'] if round_data['round'] == round_num][0]
                for agent, state in self.agent_states.items()}

class RPSTournamentSystem(InteractionAgentSystem):
    def __init__(self, agents: List[ConversableAgent]):
        system_message = """
        You are participating in a Rock-Paper-Scissors tournament.
        In each round, you will choose either 'rock', 'paper', or 'scissors'.
        Your goal is to win as many rounds as possible.
        You will be informed of the results of each round and your opponents' choices.
        Use this information to try to predict and counter your opponents' future moves.
        """
        super().__init__(
            agents=agents,
            system_message=system_message,
            action_set=["rock", "paper", "scissors"],
            observation_set=["win", "lose", "tie"],
            interaction_prompt=self._rps_interaction_prompt,
            result_parser=self._rps_result_parser,
            utility_calculator=self._rps_utility_calculator
        )

    def _rps_interaction_prompt(self, agents: List[ConversableAgent], action_set: List[str], observation_set: List[str]) -> str:
        prompt = f"Round {self.current_round}: Choose 'rock', 'paper', or 'scissors'.\n"
        for agent in agents:
            history = self.agent_states[agent.name]['history']
            if history:
                prompt += f"History for {agent.name}:\n"
                for round_info in history[-5:]:  # Show last 5 rounds
                    prompt += f"Round {round_info['round']}: Chose {round_info['action']}, Result: {round_info['utility']}\n"
        return prompt

    def _rps_result_parser(self, actions: Dict[str, str]) -> Dict[str, str]:
        parsed_actions = {}
        for agent, response in actions.items():
            for action in self.action_set:
                if action in response.lower():
                    parsed_actions[agent] = action
                    break
            if agent not in parsed_actions:
                parsed_actions[agent] = random.choice(self.action_set)
        return parsed_actions

    def _rps_utility_calculator(self, actions: Dict[str, str]) -> Dict[str, float]:
        results = {agent: 0 for agent in actions}
        agents = list(actions.keys())
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                winner = self._rps_winner(actions[agents[i]], actions[agents[j]])
                if winner == 1:
                    results[agents[i]] += 1
                elif winner == 2:
                    results[agents[j]] += 1
        return results

    def _rps_winner(self, action1: str, action2: str) -> int:
        if action1 == action2:
            return 0
        if (action1 == "rock" and action2 == "scissors") or \
           (action1 == "scissors" and action2 == "paper") or \
           (action1 == "paper" and action2 == "rock"):
            return 1
        return 2

    def get_win_rates(self) -> Dict[str, float]:
        win_rates = {}
        for agent, state in self.agent_states.items():
            total_games = len(state['history'])
            wins = sum(1 for round_info in state['history'] if round_info['utility'] > 0)
            win_rates[agent] = wins / total_games if total_games > 0 else 0
        return win_rates

# Example usage:
agents = [
    ConversableAgent(name="Agent1", system_message="You are a strategic RPS player."),
    ConversableAgent(name="Agent2", system_message="You are an unpredictable RPS player."),
    ConversableAgent(name="Agent3", system_message="You are an adaptive RPS player.")
]

tournament = RPSTournamentSystem(agents)
results = tournament.run_simulation(num_rounds=100)

print("Tournament Results:")
win_rates = tournament.get_win_rates()
for agent, win_rate in win_rates.items():
    print(f"{agent}: Win Rate: {win_rate:.2f}, Total Utility: {results[agent]}")