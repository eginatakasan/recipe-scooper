# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from matplotlib import pyplot as plt
import numpy as np
import json

# Load from a local file
with open('data/labelled.jsonl', 'r') as file:
    data = json.load(file)
    
print(data)


# %%
