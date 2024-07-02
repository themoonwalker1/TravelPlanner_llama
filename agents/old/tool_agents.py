import importlib
import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any
from pandas import DataFrame
from tqdm import tqdm
from datasets import load_dataset
import requests
from prompts import zeroshot_react_agent_prompt

# Update system path to include necessary directories
sys.path.extend([
    os.path.abspath(os.path.join(os.getcwd(), "..")),
    os.path.abspath(os.path.join(os.getcwd(), "tools/planner")),
    os.path.abspath(os.path.join(os.getcwd(), "../tools/planner"))
])

# Set environment variables
os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

# Action mapping dictionary to map action types to tool names
action_mapping = {
    "FlightSearch": "flights",
    "AttractionSearch": "attractions",
    "GoogleDistanceMatrix": "googleDistanceMatrix",
    "AccommodationSearch": "accommodation",
    "RestaurantSearch": "restaurants",
    "Planner": "planner",
    "NotebookWrite": "notebook",
    "CitySearch": "cities"
}


# Custom exceptions for handling specific errors
class CityError(Exception):
    pass


class DateError(Exception):
    pass


# Llama3 class to interact with the Llama model API
class Llama3:
    def __init__(self, llama_url: str, model: str, stream: bool, output: str, messages: List[Dict[str, Any]]):
        self.llama_url = llama_url
        self.model = model
        self.stream = stream
        self.output = output
        self.messages = messages

    def add_message(self, role: str, content: str):
        """Add a message to the list of messages to be sent to the Llama model."""
        if role not in ['system', 'user', 'assistant']:
            raise ValueError("Invalid role")
        print(content)
        self.messages.append({"role": role, "content": content})

    def send_query(self) -> Dict[str, Any]:
        """Send the query to the Llama model and return the response."""
        request = {
            "model": self.model,
            "messages": self.messages[:],
            "stream": self.stream
        }
        # request["messages"][-1]['content'] =  + request["messages"][-1]['content']
        response = requests.post(self.llama_url, json=request)
        response_json = response.json()
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, ensure_ascii=False, indent=4)
        
        self.add_message("assistant", response_json['message']['content'])
        return response_json


# ReactAgent class to manage the interaction process
class ReactAgent:
    def __init__(self, args, mode: str, tools: List[str], max_steps: int, max_retries: int,
                 illegal_early_stop_patience: int,
                 react_llm_name: str, planner_llm_name: str, city_file_path: str, agent_prompt: str):
        self.max_steps = max_steps
        self.mode = mode
        self.react_name = react_llm_name
        self.planner_name = planner_llm_name
        self.agent_prompt = agent_prompt
        self.illegal_early_stop_patience = illegal_early_stop_patience
        self.max_retries = max_retries
        self.city_set = self.load_city(city_file_path)
        self.tools = self.load_tools(tools)
        self.tools_list = tools
        self.llm = Llama3(llama_url=args.llama_url, model=react_llm_name, stream=args.stream, output=os.path.join(args.output_dir, "output.json"),
                          messages=[])
        self.__reset_agent()

    def run(self, query: str, reset: bool = True):
        """Run the agent with a given query."""
        self.query = query
        if reset:
            self.__reset_agent()
        self.llm.add_message("system", zeroshot_react_agent_prompt.format(query=self.query, scratchpad=""))

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self):
        """Perform a single step in the agent's reasoning process."""
        self.json_log.append({"step": self.step_n, "thought": "", "action": "", "observation": "", "state": ""})
        thought = self.prompt_agent(f"Give me thought number {self.step_n} and that only (without extra dialogue) in the following example format:\n\nThought {self.step_n}: [reasoning inserted here]\n")
        self.json_log[-1]['thought'] = thought
        action = self.prompt_agent(f"Give me action number {self.step_n} and that only (without extra dialogue) in the following example format:\n\nAction {self.step_n}: \nActionName[Required Information]\n")
        self.json_log[-1]['action'] = action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()
        self.last_actions.append(action)

        if len(self.last_actions) == 3:
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return

        self.scratchpad = f'Observation {self.step_n}: '
        action_type, action_arg = self.parse_action(action)
        if action_type:
            self.handle_action(action_type, action_arg)
        self.json_log[-1]['observation'] = self.current_observation
        self.step_n += 1

        if action_type == 'Planner' and self.retry_record['planner'] == 0:
            self.finished = True
            self.answer = self.current_observation

        self.llm.add_message('user', self.current_observation)

        # print(self.llm.messages[-5:])

    def prompt_agent(self, message: str) -> str:
        """Prompt the agent with a message and return the response."""
        self.llm.add_message("user", message)#"\nImportant Information Stored in Notebook:" + str(self.tools['notebook'].list_all()) + "\nMake sure to look at the conversation history and Notebook to not repeat previous steps and double check accuracy before you give an answer to the following. If you successfully got information from an action, then make sure to always write it in the notebook. Only give me the following next step:\n" + message)        
        response = self.llm.send_query()
        return response['message']['content']

    def __reset_agent(self):
        """Reset the agent's state."""
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad = ''
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []
        self.llm.messages = []
        self.retry_record = {key: 0 for key in action_mapping.values()}
        self.retry_record['invalidAction'] = 0
        self.tools = self.load_tools(self.tools_list)


    def load_tools(self, tools: List[str]) -> Dict[str, Any]:
        """Load the tools specified in the tools list."""
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module(f"tools.{tool_name}.apis")
            tool_class = getattr(module, tool_name[0].upper() + tool_name[1:])
            tools_map[tool_name] = tool_class()
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        """Load the list of valid cities from a file."""
        with open(city_set_path, 'r') as file:
            return file.read().strip().split('\n')

    def parse_action(self, action_str: str):
        """Parse the action string to extract the action type and arguments."""
        pattern = r'(\w+)\[(.+)]'
        match = re.search(pattern, action_str, re.M)
        if match:
            return match.group(1), match.group(2)
        return None, None



    def handle_action(self, action_type: str, action_arg: str):
        """Handle the action based on its type."""
        if action_type in action_mapping:
            try:
                action_func = getattr(self, f'handle_{action_type.lower()}')
                action_func(action_arg)
            except Exception as e:
                self.current_observation = f'Error in {action_type}: {str(e)}'
                self.json_log[-1]['state'] = 'Error'

    def handle_flightsearch(self, args: str):
        """Handle the FlightSearch action."""
        from_city, to_city, date = args.split(', ')
        if not validate_date_format(date):
            raise DateError(f"Invalid date format: {date}")
        if from_city not in self.city_set or to_city not in self.city_set:
            raise CityError(f"Invalid cities: {from_city}, {to_city}")
        self.current_data = self.tools['flights'].run(from_city, to_city, date)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_attractionsearch(self, args: str):
        """Handle the AttractionSearch action."""
        city = args.strip()
        if city not in self.city_set:
            raise CityError(f"Invalid city: {city}")
        self.current_data = self.tools['attractions'].run(city)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_accommodationsearch(self, args: str):
        """Handle the AccommodationSearch action."""
        city = args.strip()
        if city not in self.city_set:
            raise CityError(f"Invalid city: {city}")
        self.current_data = self.tools['accommodations'].run(city)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_restaurantsearch(self, args: str):
        """Handle the RestaurantSearch action."""
        city = args.strip()
        if city not in self.city_set:
            raise CityError(f"Invalid city: {city}")
        self.current_data = self.tools['restaurants'].run(city)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_citysearch(self, args: str):
        """Handle the CitySearch action."""
        state = args.strip()
        self.current_data = self.tools['cities'].run(state)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_googledistancematrix(self, args: str):
        """Handle the GoogleDistanceMatrix action."""
        origin, destination, mode = args.split(', ')
        self.current_data = self.tools['googleDistanceMatrix'].run(origin, destination, mode)
        self.current_observation = to_string(self.current_data)
        self.json_log[-1]['state'] = 'Successful'

    def handle_notebookwrite(self, args: str):
        """Handle the NotebookWrite action."""
        print("writing to notebook")
        self.current_observation = str(self.tools['notebook'].write(self.current_data, args))
        self.json_log[-1]['state'] = 'Successful'

    def handle_planner(self, args: str):
        """Handle the Planner action."""
        self.current_observation = str(self.tools['planner'].run(str(self.tools['notebook'].list_all()), args))
        self.answer = self.current_observation
        self.json_log[-1]['state'] = 'Successful'
    def is_finished(self) -> bool:
        """Check if the agent has finished its process."""
        return self.finished

    def is_halted(self) -> bool:
        """Check if the agent has halted due to reaching maximum steps or token length."""
        return (self.step_n > self.max_steps) and not self.finished


def validate_date_format(date_str: str) -> bool:
    """Validate the date format to ensure it matches YYYY-MM-DD."""
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))


def to_string(data) -> str:
    """Convert data to a string format, handling different data types."""
    if data is not None:
        if isinstance(data, DataFrame):
            return data.to_string(index=False)
        return str(data)
    return "None"


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_url", default="http://localhost:11434/api/chat", type=str)
    parser.add_argument("--model_name", default="l70", type=str) # crp:latest
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--stream", default=False, action='store_true')
    parser.add_argument("--set_type", default="validation", type=str)
    args = parser.parse_args()

    # Load the dataset based on the set type
    dataset = load_dataset('osunlp/TravelPlanner', args.set_type)[args.set_type]
    tools_list = ["notebook", "flights", "attractions", "accommodations", "restaurants", "googleDistanceMatrix",
                  "planner", "cities"]

    # Initialize the ReactAgent
    agent = ReactAgent(args, mode='zero_shot', tools=tools_list, max_steps=15, max_retries=3,
                       illegal_early_stop_patience=3,
                       react_llm_name=args.model_name, planner_llm_name=args.model_name,
                       city_file_path='../database/background/citySet.txt',
                       agent_prompt=zeroshot_react_agent_prompt)

    # Create output directory if it doesn't exist
    output_path = os.path.join(args.output_dir, args.set_type)
    os.makedirs(output_path, exist_ok=True)

    # Process each query in the dataset
    for number, data in enumerate(tqdm(dataset), start=1):
        if number > 1: continue
        query = data['query']
        output_file = os.path.join(output_path, f'generated_plan_{number}.json')

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result = json.load(f)
        else:
            result = [{}]

        # Run the agent to get the results
        planner_results, scratchpad, action_log = agent.run(query)
        if planner_results == 'Max Token Length Exceeded.':
            result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
            result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
            action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
        else:
            result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
            result[-1][f'{args.model_name}_two-stage_results'] = planner_results
            result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

        # Save the results to a file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
