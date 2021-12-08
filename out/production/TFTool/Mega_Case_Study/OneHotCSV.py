#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:22:51 2021

@author: overlordx

This Class makes a hoe hot table for DBPR real estate licensee data
"""
import numpy
import pandas

dataset = pandas.read_csv('resources/RealEstateSchoolLicense_mod1.csv')
numCols = dataset.columns.max()
data_frame = pandas.DataFrame(numpy.zeros(len(dataset)))


uniqueColValues = data_frame.columns['Zip']
print(uniqueColValues)
