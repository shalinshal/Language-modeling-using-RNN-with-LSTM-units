#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: assgin.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


def assign_weight_count_all_0_case_1(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence

    Input node only receives digit '0' and all the gates are
    always open.

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = [[100.] if i == 0 else [0.] for i in range(10)]
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    param_dict['wix'] = np.zeros((in_dim, out_dim))
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] =  100. * np.ones((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

def assign_weight_count_all_case_2(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence
    Input node receives all the digits '0' but input gate only
    opens for digit '0'. Other gates are always open.
    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = np.zeros((in_dim, out_dim))
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = 100. * np.ones((1, out_dim))

    param_dict['wix'] = [[100.] if i == 0 else [-100.] for i in range(10)]
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] =  np.zeros((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])
        
def assign_weight_count_all_0_after_2(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence after receiving 2
    Input node receives all the digits '0' but input gate only
    opens for digit '0'. Other gates are always open.
    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {} 
    wgx = np.zeros(shape=(10, 2))
    wgx[0, 0] = 100  
    wgx[2, 1] = 100  
    #print(wgx)
    param_dict['wgx'] = wgx
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    
    wix = -100. * np.ones(shape=(10, 2))
    wix[2, 1] = 100  
    #print(wix)
    param_dict['wix'] = wix
    wih = np.zeros(shape=(2, 1))
    wih[1] = [150]
    param_dict['wih'] = wih
    param_dict['bi'] = np.zeros((1, out_dim))

   
    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))  

   
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))  

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])



def assign_weight_count_all_0_after_2_with_reset(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence after receiving 2
    Input node receives all the digits '0' but input gate only
    opens for digit '0'. Other gates are always open.
    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    wgx = np.zeros(shape=(10, 2))
    wgx[0, 0] = 100
    wgx[2, 1] = 100
    param_dict['wgx'] = wgx
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))
    wix = -100. * np.ones(shape=(10, 2))
    wix[2, 1] = 100
    param_dict['wix'] = wix
    wih = np.zeros(shape=(2, 1))
    wih[1] = [150]
    param_dict['wih'] = wih
    param_dict['bi'] = np.zeros((1, out_dim))
    
    param_dict['wfx'] = [[-100.] if i==3 else [100.] for i in range(10)]
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = np.zeros((1, out_dim))
    
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))  

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])
