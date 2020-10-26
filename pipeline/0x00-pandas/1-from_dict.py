#!/usr/bin/env python3
"""Function that creates a pd.DataFrame from a dictionary"""
import pandas as pd

data = {"A":{"First":'0.0', "Second": "one"},
        "B":{"First":'0.5', "Second": "two"},
        "C":{"First":'1.0', "Second": "three"},
        "D":{"First":'1.5', "Second": "four"}
        }

df = pd.DataFrame(data).transpose()
