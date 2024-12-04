# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:08:32 2024

@author: Noa

A custom parameter for the PNPS with added dispersion
scan_no refers to which scan the corresponding spectrum belongs to
(each scan has different amounts of added dispersion)
"""
import numpy as np

class Param_obj:
    def __init__(self,delay,scan_no):
        self.delay = delay
        self.scan_no = int(scan_no)

    def __hash__(self):
        return hash((self.delay, self.scan_no))

    def __eq__(self, other):
        return (self.delay, self.scan_no) == (other.delay, other.scan_no)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
    
def get_delays_array(param):
    # takes an array of Param_obj's and extracts an array of just the delays
    delays_array = np.ndarray(param.shape)
    for i, p in enumerate(param):
        delays_array[i] = p.delay
    return delays_array

def get_scan_num_array(param):
    # takes an array of Param_obj's and extracts an array of just the delays
    scan_num_array = np.ndarray(param.shape, dtype=int)
    for i, p in enumerate(param):
        scan_num_array[i] = p.scan_no
    return scan_num_array

def create_param_array(delays, scan_no):
    # takes an array of delays and scan numbers and combines them into an array
    # of Param_obj's
    param_array = np.ndarray(delays.shape, dtype=object)
    for i in range(delays.size):
        param_array[i] = Param_obj(delays[i], scan_no[i])
    return param_array