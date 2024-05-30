import re
import string
import os
import sys
import json
import time

import requests
from typing import List, Dict, Any
from pandas import DataFrame
from datetime import datetime
import tiktoken
from tqdm import tqdm
import argparse
from datasets import load_dataset
from prompts import zeroshot_react_agent_prompt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
import pandas as pd

pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

actionMapping = {"FlightSearch":"flights","AttractionSearch":"attractions","GoogleDistanceMatrix":"googleDistanceMatrix","AccommodationSearch":"accommodation","RestaurantSearch":"restaurants","Planner":"planner","NotebookWrite":"notebook","CitySearch":"cities"}


class CityError(Exception):
    pass

class DateError(Exception):
    pass

class Llama3:
    def __init__(self,
                 llama_url="http://localhost:11434/api/chat",
                 model="llama3",
                 stream=False,
                 output="./test_output.json",
                 messages=[]):
        self.llama_url = llama_url
        self.model = model
        self.stream = stream
        self.output = output
        self.messages = messages

    def add_message(self, role, content):
        if role not in ['user', 'assistant']:
            raise RuntimeError("Invalid role")
        self.messages.append({"role": role, "content": content})

    def send_query(self):
        request = {
            "model": self.model,
            "messages": self.messages,
            "stream": self.stream
        }

        response = requests.post(self.llama_url, json=request)
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)

        return response.json()


class ReactAgent:
    def __init__(self,
                 args,
                 mode: str = 'zero_shot',
                 tools: List[str] = None,
                 max_steps: int = 30,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name='llama3',
                 planner_llm_name='llama3',
                 city_file_path='../database/background/citySet.txt',
                 agent_prompt=zeroshot_react_agent_prompt) -> None:

        self.answer = ''
        self.max_steps = max_steps
        self.mode = mode
        self.react_name = react_llm_name
        self.planner_name = planner_llm_name
        self.agent_prompt = agent_prompt

        self.json_log = []
        self.current_observation = ''
        self.current_data = None

        self.max_token_length = 8000
        self.llm = Llama3()

        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        self.last_actions = []
        self.city_set = self.load_city(city_file_path)

        self.__reset_agent()

    def run(self, query, reset=True) -> None:
        self.query = query

        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:
        self.json_log.append({"step": self.step_n, "thought": "", "action": "", "observation": "", "state": ""})

        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()

        print(self.scratchpad.split('\n')[-1])
        self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'\nThought {self.step_n}:', "")

        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()

        if action is None or action == '' or action == '\n':
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += ' ' + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        self.last_actions.append(action)
        self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:', "")

        if len(self.last_actions) == 3:
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return

        self.scratchpad += f'\nObservation {self.step_n}: '

        if action is None or action == '' or action == '\n':
            action_type = None
            action_arg = None
            self.scratchpad += "No feedback from the environment due to the null action."
        else:
            action_type, action_arg = self.parse_action(action)

            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                else:
                    pending_action = 'invalidAction'

                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"invalidAction early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

            self.handle_action(action_type, action_arg)

        if action is None or action == '' or action == '\n':
            print(f'Observation {self.step_n}: ' + "No feedback from the environment due to the null action.")
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
        else:
            print(f'Observation {self.step_n}: ' + self.current_observation + '\n')
            self.json_log[-1]['observation'] = self.current_observation

        self.step_n += 1

        if action_type and action_type == 'Planner' and self.retry_record['planner'] == 0:
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def prompt_agent(self) -> str:
        while True:
            self.llm.add_message("user", self._build_agent_prompt())
            response = self.llm.send_query()
            request = response['choices'][0]['message']['content']
            return request

    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (self.step_n > self.max_steps or len(self.scratchpad) > self.max_token_length) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad = ''
        self.__reset_record()
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []

        if 'notebook' in self.tools:
            self.tools['notebook'].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))
            tools_map[tool_name] = getattr(module, tool_name[0].upper() + tool_name[1:])()
            if tool_name == 'planner' and planner_model_name is not None:
                tools_map[tool_name] = getattr(module, tool_name[0].upper() + tool_name[1:])(model_name=planner_model_name)
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, 'r').read().strip().split('\n')
        for unit in lines:
            city_set.append(unit)
        return city_set

    def parse_action(self, string: str):
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        return None, None

    def handle_action(self, action_type: str, action_arg: str) -> None:
        if action_type == 'FlightSearch':
            try:
                if validate_date_format(action_arg.split(', ')[2]) and validate_city_format(action_arg.split(', ')[0], self.city_set) and validate_city_format(action_arg.split(', ')[1], self.city_set):
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(), 'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools['flights'].run(action_arg.split(', ')[0], action_arg.split(', ')[1], action_arg.split(', ')[2])
                    self.current_observation = str(to_string(self.current_data))
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = 'Successful'
            except DateError:
                self.retry_record['flights'] += 1
                self.current_observation = f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                self.scratchpad += f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                self.json_log[-1]['state'] = 'Illegal args. DateError'
            except ValueError as e:
                self.retry_record['flights'] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]['state'] = 'Illegal args. City Error'
            except Exception as e:
                print(e)
                self.retry_record['flights'] += 1
                self.current_observation = 'Illegal Flight Search. Please try again.'
                self.scratchpad += 'Illegal Flight Search. Please try again.'
                self.json_log[-1]['state'] = 'Illegal args. Other Error'

        elif action_type == 'AttractionSearch':
            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(), 'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools['attractions'].run(action_arg)
                    self.current_observation = to_string(self.current_data).strip('\n').strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = 'Successful'
            except ValueError as e:
                self.retry_record['attractions'] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]['state'] = 'Illegal args. City Error'
            except Exception as e:
                print(e)
                self.retry_record['attractions'] += 1
                self.current_observation = 'Illegal Attraction Search. Please try again.'
                self.scratchpad += 'Illegal Attraction Search. Please try again.'
                self.json_log[-1]['state'] = 'Illegal args. Other Error'

        elif action_type == 'AccommodationSearch':
            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(), 'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools['accommodations'].run(action_arg)
                    self.current_observation = to_string(self.current_data).strip('\n').strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = 'Successful'
            except ValueError as e:
                self.retry_record['accommodations'] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]['state'] = 'Illegal args. City Error'
            except Exception as e:
                print(e)
                self.retry_record['accommodations'] += 1
                self.current_observation = 'Illegal Accommodation Search. Please try again.'
                self.scratchpad += 'Illegal Accommodation Search. Please try again.'
                self.json_log[-1]['state'] = 'Illegal args. Other Error'

        elif action_type == 'RestaurantSearch':
            try:
                if validate_city_format(action_arg, self.city_set):
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(), 'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools['restaurants'].run(action_arg)
                    self.current_observation = to_string(self.current_data).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = 'Successful'
            except ValueError as e:
                self.retry_record['restaurants'] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]['state'] = 'Illegal args. City Error'
            except Exception as e:
                print(e)
                self.retry_record['restaurants'] += 1
                self.current_observation = 'Illegal Restaurant Search. Please try again.'
                self.scratchpad += 'Illegal Restaurant Search. Please try again.'
                self.json_log[-1]['state'] = 'Illegal args. Other Error'

        elif action_type == "CitySearch":
            try:
                self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(), 'Masked due to limited length. Make sure the data has been written in Notebook.')
                self.current_observation = to_string(self.tools['cities'].run(action_arg)).strip()
                self.scratchpad += self.current_observation
                self.__reset_record()
                self.json_log[-1]['state'] = 'Successful'
            except ValueError as e:
                self.retry_record['cities'] += 1
                self.current_observation = str(e)
                self.scratchpad += str(e)
                self.json_log[-1]['state'] = 'Illegal args. State Error'
            except Exception as e:
                print(e)
                self.retry_record['cities'] += 1
                self.current_observation = 'Illegal City Search. Please try again.'
                self.scratchpad += 'Illegal City Search. Please try again.'
                self.json_log[-1]['state'] = 'Illegal args. Other Error'

        elif action_type == 'GoogleDistanceMatrix':
            try:
                self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(), 'Masked due to limited length. Make sure the data has been written in Notebook.')
                self.current_data = self.tools['googleDistanceMatrix'].run(action_arg.split(', ')[0], action_arg.split(', ')[1], action_arg.split(', ')[2])
                self.current_observation = to_string(self.current_data)
                self.scratchpad += self.current_observation
                self.__reset_record()
                self.json_log[-1]['state'] = 'Successful'
            except Exception as e:
                print(e)
                self.retry_record['googleDistanceMatrix'] += 1
                self.current_observation = 'Illegal GoogleDistanceMatrix. Please try again.'
                self.scratchpad += 'Illegal GoogleDistanceMatrix. Please try again.'
                self.json_log[-1]['state'] = 'Illegal args. Other Error'

        elif action_type == 'NotebookWrite':
            try:
                self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(), 'Masked due to limited length. Make sure the data has been written in Notebook.')
                self.current_observation = str(self.tools['notebook'].write(self.current_data, action_arg))
                self.scratchpad += self.current_observation
                self.__reset_record()
                self.json_log[-1]['state'] = 'Successful'
            except Exception as e:
                print(e)
                self.retry_record['notebook'] += 1
                self.current_observation = f'{e}'
                self.scratchpad += f'{e}'
                self.json_log[-1]['state'] = 'Illegal args. Other Error'

        elif action_type == "Planner":
            self.current_observation = str(self.tools['planner'].run(str(self.tools['notebook'].list_all()), action_arg))
            self.scratchpad += self.current_observation
            self.answer = self.current_observation
            self.__reset_record()
            self.json_log[-1]['state'] = 'Successful'
        else:
            self.retry_record['invalidAction'] += 1
            self.current_observation = 'Invalid Action. Valid Actions are FlightSearch[Departure City, Destination City, Date] / AccommodationSearch[City] / RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query].'
            self.scratchpad += self.current_observation
            self.json_log[-1]['state'] = 'invalidAction'

def validate_date_format(date_str: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(pattern, date_str):
        raise DateError
    return True

def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not valid city in {str(city_set)}.")
    return True

def to_string(data) -> str:
    if data is not None:
        if isinstance(data, DataFrame):
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)

if __name__ == '__main__':
    tools_list = ["notebook", "flights", "attractions", "accommodations", "restaurants", "googleDistanceMatrix", "planner", "cities"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="llama3")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    if args.set_type == 'validation':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    elif args.set_type == 'test':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'test')['test']

    print(query_data_list)
    numbers = [i for i in range(1, len(query_data_list) + 1)]
    agent = ReactAgent(None, tools=tools_list, max_steps=30, react_llm_name=args.model_name, planner_llm_name=args.model_name)

    for number in tqdm(numbers[:]):
        query = query_data_list[number - 1]['query']

        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
            os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))

        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
            result = [{}]
        else:
            result = json.load(open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))

        while True:
            planner_results, scratchpad, action_log = agent.run(query)
            if planner_results is not None:
                break

        if planner_results == 'Max Token Length Exceeded.':
            result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
            result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
            action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
            result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
        else:
            result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad
            result[-1][f'{args.model_name}_two-stage_results'] = planner_results
            result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

        with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
            json.dump(result, f, indent=4)
