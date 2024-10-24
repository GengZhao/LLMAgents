from typing import List, Dict, Callable, Optional, Any
from crewai import Agent, Task, Process, Crew, LLM
import logging
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class InteractionState:
    """Represents the current state of the interaction system."""
    round: int
    global_state: Dict[str, Any]
    agent_states: Dict[str, Dict[str, Any]]
    history: List[Dict[str, Any]]

class InteractionAgentSystem:
    def __init__(
        self,
        agents: List[Agent],
        system_message: str,
        action_set: List[str],
        observation_set: List[str],
        interaction_function: Callable[[Dict[str, str], InteractionState], tuple[Dict[str, str], Dict[str, Any]]],
        max_rounds: int = 10,
        logging_level: int = logging.INFO
    ):
        """
        Initialize the interaction system.
        
        Args:
            agents: List of Crew AI agents
            system_message: Context message describing the interaction setup
            action_set: List of allowed actions
            observation_set: List of possible observations
            interaction_function: Function that processes agents' actions and returns new observations
            max_rounds: Maximum number of interaction rounds
            logging_level: Logging verbosity level
        """
        self.agents = {agent.role: agent for agent in agents}
        self.system_message = system_message
        self.action_set = set(action_set)
        self.observation_set = set(observation_set)
        self.interaction_function = interaction_function
        self.max_rounds = max_rounds
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
        
        # Initialize state
        self.state = InteractionState(
            round=0,
            global_state={},
            agent_states={agent.role: {} for agent in agents},
            history=[]
        )
        
        # Create crew instance
        self.crew = Crew(
            agents=agents,
            process=Process.sequential,
            verbose=logging_level == logging.DEBUG,
        )

    def _validate_action(self, action: str) -> bool:
        """Validate if an action is in the allowed action set."""
        return action in self.action_set

    def _create_agent_prompt(self, agent_role: str, observation: str) -> str:
        """Create the prompt for an agent given their current observation."""
        history_summary = self._get_history_summary(agent_role)
        
        # prompt = f"""
        # {self.system_message}

        # Current Round: {self.state.round + 1}/{self.max_rounds}
        # Your Role: {agent_role}
        
        # Historical Context:
        # {history_summary}
        
        # Current Observation:
        # {observation}
        
        # Available Actions:
        # {', '.join(self.action_set)}
        
        # Please analyze the situation and choose an action from the available set.
        # Explain your reasoning step by step, then conclude with your chosen action.
        
        # Your response should be structured as:
        # Reasoning: [Your thought process]
        # Action: [Your chosen action]
        # """
        prompt = f"""
        {self.system_message}

        Current Round: {self.state.round + 1}/{self.max_rounds}
        Your Role: {agent_role}
        
        Historical Context:
        {history_summary}
        
        Current Observation:
        {observation}
        
        Available Actions:
        {', '.join(self.action_set)}
        
        Please analyze the situation and choose an action from the available set.
        
        Your response should be structured as:
        Reasoning: [Your thought process - Limited to 150 characters]
        Action: [Your chosen action]
        """
        # Explain your reasoning step by step, then conclude with your chosen action.
        return prompt

    def _get_history_summary(self, agent_role: str, max_rounds: int = 2) -> str:
        """Generate a summary of recent interaction history relevant to the agent."""
        if not self.state.history:
            return "No previous interactions."
            
        recent_history = self.state.history[-max_rounds:]
        summary = []
        
        for round_data in recent_history:
            round_summary = f"Round {round_data['round']}:\n"
            for role, actions in round_data['actions'].items():
                if role == agent_role or round_data['public']:
                    round_summary += f"- {role}: {actions}\n"
            summary.append(round_summary)
            
        return "\n".join(summary)

    def _process_agent_response(self, response: str) -> str:
        """Extract the action from the agent's response."""
        try:
            # Find the action line
            action_line = [line for line in response.split('\n') 
                         if line.strip().lower().startswith('action:')][-1]
            action = action_line.split(':', 1)[1].strip()
            
            if not self._validate_action(action):
                self.logger.warning(f"Invalid action received: {action}")
                return None
                
            return action
        except Exception as e:
            self.logger.error(f"Error processing agent response: {e}")
            return None

    def run_round(self, current_observations: Dict[str, str]) -> Dict[str, str]:
        """Run a single round of interaction."""
        # Create tasks for each agent
        tasks = []
        crews = []
        
        for role, agent in self.agents.items():
            task = Task(
                description=self._create_agent_prompt(role, current_observations.get(role, "")),
                expected_output="Chosen action based on the situation",
                agent=agent,
            )
            tasks.append(task)
            # Create individual crews for concurrent execution
            crews.append(Crew(
                agents=[agent],
                tasks=[task],
                verbose=self.logger.level == logging.DEBUG,
            ))
            # Print summary of the task and crew
            self.logger.info(f"Task for {role}: {task.description}")

        # Kick off all crews concurrently
        responses = [crew.kickoff() for crew in crews]
        # responses = await asyncio.gather(*crew_futures)

        # Process responses into actions
        actions = {}
        for role, response in zip(self.agents.keys(), responses):
            action = self._process_agent_response(response)
            actions[role] = action if action else list(self.action_set)[0]  # Default to first action if invalid

        return actions
    
    def run_interaction(self) -> Dict[str, Any]:
        """Run the full interaction process for all agents."""
        current_observations = {role: "" for role in self.agents.keys()}

        while self.state.round < self.max_rounds:
            self.logger.info(f"Starting round {self.state.round + 1}")
            
            # Get actions from all agents
            round_actions = self.run_round(current_observations)
            
            # Process interactions
            new_observations, state_updates = self.interaction_function(
                round_actions, 
                self.state
            )
            
            # Update state
            self.state.round += 1
            self.state.global_state.update(state_updates.get('global', {}))
            for role, updates in state_updates.get('agents', {}).items():
                self.state.agent_states[role].update(updates)
            
            # Record history
            self.state.history.append({
                'round': self.state.round,
                'timestamp': datetime.now().isoformat(),
                'actions': round_actions,
                'observations': new_observations,
                'public': state_updates.get('public', True)
            })
            
            # Update observations for next round
            current_observations = new_observations
            
            self.logger.info(f"Completed round {self.state.round}")
            
        return {
            'final_state': self.state.global_state,
            'agent_states': self.state.agent_states,
            'history': self.state.history
        }

    def get_interaction_summary(self) -> str:
        """Generate a summary of the entire interaction."""
        summary = [f"Interaction Summary ({self.state.round} rounds)"]
        
        for round_data in self.state.history:
            round_summary = f"\nRound {round_data['round']}:"
            for role, action in round_data['actions'].items():
                round_summary += f"\n- {role}: {action}"
            summary.append(round_summary)
            
        return "\n".join(summary)