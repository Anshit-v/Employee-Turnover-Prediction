# -*- coding: utf-8 -*-
""" Data Information """


def dataInfo(a):
    print(a.head())
    
    print("/n the shape of dataset is /n",a.shape)
    
    print("/n Info.  regarding dataset /n",a.info())
    
    #to find the range of tuples in  attributes and other info.
    print("/n the range of tuples in  attributes /n ",a.describe())