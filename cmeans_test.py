# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:13:39 2019

@author: Sehgals
"""

import numpy as np
import logging

import sys
sys.path.append('..')


from fuzzy_clustering import  FCM
from fuzzycmeans.visualization import draw_model_2d


# X = np.array([[1, 1], [1, 2], [2, 2], [9, 10], [10, 10], [10, 9], [9, 9], [20,20]])
X = np.random.randint(20, size=(200, 2))
fcm = FCM(n_clusters=3)
fcm.set_logger(tostdout=True, level=logging.DEBUG)
classes = np.random.randint(3, size=200)
fcm.fit(X, classes) #[0, 0, 0, 1, 1, 1, 1, 2]
# fcm.fit(X)
testing_data = np.random.randint(20, size=(50, 2))
# testing_data = np.array([[0, 1.9], [5, 3], [4, 4], [8, 9], [9.5, 6.5], [5, 5], [15,15], [12,12], [14,14], [19,10]])
predicted_membership = fcm.predict(testing_data)
np.savetxt('test_data_membership.csv', predicted_membership, delimiter=',')
# print ("\n\ntesting data")
# print (testing_data)
print ("predicted membership")
print (predicted_membership)
print ("\n\n")
draw_model_2d(fcm, data=testing_data, membership=predicted_membership)