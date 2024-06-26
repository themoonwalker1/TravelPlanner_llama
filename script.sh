#!/bin/bash

# Check if the user has provided an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

# Assign the argument to a variable
model=$1

cd agents
../venv/bin/python3 tool_agents.py --model_name $model
cd ..
./venv/bin/python3 postprocess/parsing.py --model_name $model
./venv/bin/python3 postprocess/element_extraction.py --model_name $model
./venv/bin/python3 postprocess/combination.py --model_name $model
./venv/bin/python3 evaluation/eval.py 


