from __future__ import division, print_function, absolute_import
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module defines base classes for all models.  The base class of all
models is `~astropy.modeling.Model`. `~astropy.modeling.FittableModel` is
the base class for all fittable models. Fittable models can be linear or
nonlinear in a regression analysis sense.

All models provide a `__call__` method which performs the transformation in
a purely mathematical way, i.e. the models are unitless.  Model instances can
represent either a single model, or a "model set" representing multiple copies
of the same type of model, but with potentially different values of the
parameters in each model making up the set.
"""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import operator


def _alt_model_oper(oper, **kwargs):
    """
    This is an alternate version of compound models intended to use 
    much less memory than the default.
    """
    return lambda left, right: _AltCompoundModel(oper,
            left, right, **kwargs)


class _AltCompoundModel:

    def __init__(self, op, left, right, name=None):
        self.op = op
        self.left = left
        self.right = right
        if op in ['+', '-', '*', '/', '**']:
            if (left.n_inputs != right.n_inputs) or \
               (left.n_outputs != right.n_outputs):
                raise ValueError('Both operands must match numbers of inputs and outputs')
            else:
                self.n_inputs = left.n_inputs
                self.n_outputs = left.n_outputs
        elif op == '&':
            self.n_inputs = left.n_inputs + right.n_inputs
            self.n_outputs = left.n_outputs + right.n_outputs
        elif op == '|':
            if left.n_outputs != right.n_inputs:
                raise ValueError('left operand number of outputs must match right operand number of inputs')
            self.n_inputs = left.n_inputs
            self.n_outputs = right.n_outputs
        else:
            raise ValueError('Illegal operator')        
        self.name = name

    def __call__(self, *args, **kw):
        op = self.op
        if op != '&':
            leftval = self.left(*args, **kw)
            if op != '|':
                rightval = self.right(*args, **kw)

        else:
            leftval = self.left(*(args[:self.left.n_inputs]), **kw)
            rightval = self.right(*(args[self.left.n_inputs:]), **kw)
  
        if op == '+':
            return binary_operation(operator.add, leftval, rightval)
        elif op == '-':
            return binary_operation(operator.sub, leftval, rightval)
        elif op == '*':
            return binary_operation(operator.mul, leftval, rightval)
        elif op == '/':
            return binary_operation(operator.truediv, leftval, rightval)
        elif op == '**':
            return binary_operation(operator.pow, leftval, rightval)
        elif op == '&':
            if not isinstance(leftval, tuple):
                leftval = (leftval,)
            if not isinstance(rightval, tuple):
                rightval = (rightval,)
            return leftval + rightval
        elif op == '|':
            if isinstance(leftval, tuple):
                return self.right(*leftval, **kw)
            else:
                return self.right(leftval, **kw)
        else:
            raise ValueError('unrecognized operator')

    __add__ =     _alt_model_oper('+')
    __sub__ =     _alt_model_oper('-')
    __mul__ =     _alt_model_oper('*')
    __truediv__ = _alt_model_oper('/')
    __pow__ =     _alt_model_oper('**')
    __or__ =      _alt_model_oper('|')
    __and__ =     _alt_model_oper('&')


def binary_operation(binoperator, left, right):
    '''
    Sum over all inputs in tuple, if present.
    '''
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple([binoperator(item[0], item[1]) for item in zip(left, right)])
    else:
        return binoperator(left, right)
   

