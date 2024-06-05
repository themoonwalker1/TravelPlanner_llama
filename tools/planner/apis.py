import json
import sys
import os
import re
import time
from typing import List, Union
from enum import Enum
import argparse

import requests
from datasets import load_dataset
import tiktoken
from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt, reflect_prompt, \
    react_reflect_planner_agent_prompt, REFLECTION_HEADER
from env import ReactEnv, ReactReflectEnv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))


class CityError(Exception):
    pass


class DateError(Exception):
    pass


class Llama3:
    def __init__(self,
                 llama_url="http://localhost:11434/api/chat",
                 model="llama3:8b-instruct-fp16",
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


def catch_llama3_error():
    error = sys.exc_info()[0]
    print("API error:", error)
    time.sleep(60)


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = 'llama3:8b-instruct-fp16',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.llm = Llama3()

        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n' + self._build_agent_prompt(text, query))
        if len(self.enc.encode(self._build_agent_prompt(text, query))) > 12000:
            return 'Max Token Length Exceeded.'
        else:
            self.llm.add_message("user", self._build_agent_prompt(text, query))
            response = self.llm.send_query()
            return response['message']['content']

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt.format(
            text=text,
            query=query)


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """

    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'llama3:8b-instruct-fp16',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.llm = Llama3()
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset=True) -> None:
        self.query = query
        self.text = text

        if reset:
            self.reset()

        while not (self.is_halted() or self.is_finished()):
            self.step()

        return self.answer, self.scratchpad

    def step(self) -> None:
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError(
                        'The sub plan cannot be parsed into JSON format. Please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = 'The sub plan cannot be parsed into JSON format. Please check.'
            except ValueError as e:
                observation = str(e)

        elif action_type == 'Finish':
            self.finished = True
            observation = 'The plan is finished.'
            self.answer = action_arg

        else:
            observation = f'Action {action_type} is not supported.'

        self.curr_step += 1
        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                self.llm.add_message("user", self._build_agent_prompt())
                response = self.llm.send_query()
                return format_step(response['choices'][0]['message']['content'])
            except:
                catch_llama3_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            query=self.query,
            text=self.text,
            scratchpad=self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """

    def __init__(self,
                 agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 model_name: str = 'llama3:8b-instruct-fp16',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.llm = Llama3()
        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset=True) -> None:
        self.query = query
        self.text = text

        if reset:
            self.reset()

        while not (self.is_halted() or self.is_finished()):
            self.step()
            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        return self.answer, self.scratchpad

    def step(self) -> None:
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError(
                        'The sub plan cannot be parsed into JSON format. Please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = 'The sub plan cannot be parsed into JSON format. Please check.'
            except ValueError as e:
                observation = str(e)

        elif action_type == 'Finish':
            self.finished = True
            observation = 'The plan is finished.'
            self.answer = action_arg

        else:
            observation = f'Action {action_type} is not supported.'

        self.curr_step += 1
        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_agent(self) -> str:
        while True:
            try:
                self.llm.add_message("user", self._build_agent_prompt())
                response = self.llm.send_query()
                return format_step(response['choices'][0]['message']['content'])
            except:
                catch_llama3_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def prompt_reflection(self) -> str:
        while True:
            try:
                self.llm.add_message("user", self._build_reflection_prompt())
                response = self.llm.send_query()
                return format_step(response['choices'][0]['message']['content'])
            except:
                catch_llama3_error()
                print(self._build_reflection_prompt())
                print(len(self.enc.encode(self._build_reflection_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            query=self.query,
            text=self.text,
            scratchpad=self.scratchpad,
            reflections=self.reflections_str)

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            query=self.query,
            text=self.text,
            scratchpad=self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ''
        self.env.reset()


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')


def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    if match:
        action_type = match.group(1)
        action_arg = match.group(2)
        return action_type, action_arg
    return None, None


def format_reflections(reflections: List[str],
                       header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])


if __name__ == '__main__':
    pass
