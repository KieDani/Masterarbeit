#!/bin/bash

rm -r docs
mkdir docs
pdoc3 --html --output-dir docs test.py my_sampler.py my_operators.py my_models.py plot.py helping_functions.py my_machines.py 

