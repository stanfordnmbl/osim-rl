from osim.env import ProstheticsEnv
import numpy as np
import unittest

# Source: https://stackoverflow.com/questions/39135433/how-to-flatten-nested-python-dictionaries

def flatten(d):    
    res = []  # Result list
    if isinstance(d, dict):
        for key, val in sorted(d.items()):
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d        
    else:
        res = [d]

    return res

d = { '123': { 'key3': 3, 'key2': 11, 'key1': 1 },
      '124': { 'key1': 6, 'key2': 56, 'key3': 6 },
      '125': { 'key1': 7, 'key2': 44, 'key3': 9 },
    }

env = ProstheticsEnv()
print(flatten(env.reset()))
print(flatten(d))

