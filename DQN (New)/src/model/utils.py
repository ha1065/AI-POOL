import functools
import operator

def prod(l): #returns product of state space list
    return functools.reduce(operator.mul, l, 1)
