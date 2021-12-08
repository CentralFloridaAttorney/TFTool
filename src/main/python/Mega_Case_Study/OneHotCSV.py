#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:22:51 2021

@author: overlordx

This Class makes a hoe hot table for DBPR real estate licensee data
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('resources/RE_rgn5_new2.csv')
data_frame = pd.DataFrame(dataset)
data_frame.fillna(0)
uniqueColValues = data_frame['DBA Name'].values
print(uniqueColValues)