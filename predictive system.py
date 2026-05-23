# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

input_data = (3,78,50,32,88,31,0.248,26) 

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


std_data = scaler.transform(input_data_reshaped) 

prediction = loaded_model.predict(std_data) 
print(prediction)

if prediction[0] == 0:
  print('The person is not diabetic')
else:
  print('The person is diabetic')
