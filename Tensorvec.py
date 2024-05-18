import copy
import random
import sys
from toolfunc import Ut
from Normalize import Normalizer
import numpy as np

import torch
Ut.colorWord('torch package in Tensorvec has initialized',color='g')

class TensorVec():
    def __init__(self, embList:list=None, value=None,embnum = 0,device='cpu'):
        self.device = device
        self.isFunction = embList is None and embnum==0
        if self.isFunction:
            self.embList = self.embSize = self.value = None
            return

        if embnum > 0 and embList is None:
            #self.embList = self.toGradTensor([round(random.random() * 100, 3) for _ in range(embnum)])
            self.embList = self.toGradTensor(list(np.random.normal(loc=Normalizer.mean,scale=Normalizer.std,size=embnum)))
            self.embSize = embnum

        elif not isinstance(embList,torch.Tensor):
            self.embList = self.toGradTensor(embList)
            self.embSize =  len(embList)
        else:
            self.embList = embList.to(device)
            self.embSize = list(self.embList.size())[0]

        self.value = value

    def __add__(self, other):
        if isinstance(other,TensorVec):
            if self.embSize == other.embSize:
                return TensorVec(torch.add(self.embList, other.embList),device=self.device)
        if Ut.raiseError('Wrong Input -',sys._getframe().f_code.co_name):
            return

        errorL = [self, other]
        maxL = max(len(str(Tvec)) if isinstance(Tvec,TensorVec) else -1 for Tvec in errorL)
        errorOpt = [str(Tvec) + ' '*(maxL-len(str(Tvec))+2) + f' with dimension of {len(Tvec) if isinstance(Tvec,TensorVec) else -1}' for Tvec in errorL]
        print('    input emb list: -\n        ', end='')
        Ut.colorWord('\n        '.join(errorOpt), color='g')

    def __sub__(self, other):

        if isinstance(other,TensorVec):
            if self.embSize == other.embSize:
                return TensorVec(torch.sub(self.embList, other.embList),device=self.device)
        if Ut.raiseError('Wrong dimension size', sys._getframe().f_code.co_name):
            return

        errorL = [self, other]
        maxL = max(len(str(Tvec)) if isinstance(Tvec, TensorVec) else -1 for Tvec in errorL)
        errorOpt = [str(Tvec) + ' ' * (maxL - len(
            str(Tvec)) + 2) + f' with dimension of {len(Tvec) if isinstance(Tvec, TensorVec) else -1}' for Tvec in
                    errorL]
        print('    input emb list: -\n        ', end='')
        Ut.colorWord('\n        '.join(errorOpt), color='g')

    def __len__(self):
        return self.embSize

    def pos(self, *args):
        return TensorVec(self.embList, value=None,device=self.device)

    def nestmap(self, x, func=lambda x:x, tensor=False):
        if not (tensor and isinstance(x,torch.Tensor)):
            if isinstance(x,(list,tuple)):
                return [self.nestmap(y, func) for y in x]
            if isinstance(x, torch.Tensor):
                try:
                    x = x.item()
                except:
                    x = list(x)
                return self.nestmap(x, func)
        try:
            return func(x)
        except Exception as e:
            funcattribute = func.__name__
            if Ut.raiseError(f'Error occurs during function {funcattribute} to process {x}', sys._getframe().f_code.co_name,ctn=True):
                return
            Ut.colorWord('    '+str(e), color='r')
            return

    def toGradTensor(self, x):
        return torch.tensor(self.nestmap(x, float), requires_grad=True,device=self.device)

    def getDistance(self, *args):
        other = list()
        for tv in args:
            other.append(tv)
        if self.isFunction:
            return [other[i].getDistance(other[i + 1]) for i in range(0, len(other) - 1)]
        if args:
            if len(other)>1:
                other = [self]+other
                return [other[i].getDistance(other[i + 1]) for i in range(0, len(other) - 1)]

            result = self.distanceInMultipleDimension(self, other[0])
            return result
        Ut.raiseError(f"Unknow input type of {type(other)}",
                      sys._getframe().f_code.co_name)

    def distanceInMultipleDimension(self, vec1, vec2):
        distance = 0
        differences = (vec1 - vec2)
        if isinstance(differences,TensorVec):
            for difference in differences.embList:
                distance = (difference ** 2 + distance ** 2) ** 0.5

            return distance
        Ut.raiseError(f"Need TensorVec object to calculate difference, got {type(differences)}", sys._getframe().f_code.co_name)
        return -1


    def grad(self):
        return self.embList.grad

    def __iter__(self):
        return list(self.nestmap(self.embList,lambda x:x))

    def __str__(self):
        output = str(list(self.nestmap(self.embList, lambda x:round(x, Normalizer.round))))
        if len(output)>35:
            output = output[:15]+' ... ' + output[-15:]
        return output
    def repr(self):
        variables = vars(self)
        Ut.colorWord(f'Variables of {repr(self)} -', color='b')
        maxL = max(map(len,variables.keys()))
        for k in variables.keys():
            outputLine = f'        key : {k}'+' '*(maxL-len(k)+3)+f'value : {variables[k]}'
            Ut.colorWord(outputLine, color='b')

    def parameters(self):
        return self.embList

funcs = TensorVec()

def Vec(embList=None, value=None,embnum=0,device='cpu'):
    if isinstance(embList,list):
        if isinstance(embList[0],list):
            if not all(isinstance(e, list) or isinstance(e, float) or isinstance(e, int) for e in embList):
                Ut.raiseError('Wrong type of vec',sys._getframe().f_code.co_name)
                return
            return [Vec(embList=embl, value=v,embnum=nm,device=device) for embl, v,nm in zip(embList, value,embnum)] \
                if value is not None else [Vec(embList=embl,device=device) for embl in embList]
        return TensorVec(embList, value,embnum,device=device)
    elif isinstance(embList,TensorVec):
        return copy.copy(embList)

    elif isinstance(embList,torch.Tensor):
        return TensorVec(funcs.nestmap(embList), value, embnum,device=device)

    elif embList is None and embnum != 0:
        return TensorVec(embList, value, embnum,device=device)

    Ut.raiseError('Wrong type of vec',sys._getframe().f_code.co_name)

    return

if __name__ == '__main__':
    Ut.setWarning(False)
    a = Vec(value='a',embnum=3,device='cuda')
    b = Vec([0,0,0],value='b',device='cuda')
    a.repr()
    #a+1
    #print(a.device,a.embList.device,a.pos().embList.device)

    dist = a.pos().getDistance(b)
    dist2 = funcs.getDistance(a,b)

    dist.backward()
    print(dist)
    print(a.grad(),b.grad())
    print(dist2)
    '''
    Expect output
    
    Variables of <__main__.TensorVec object at 0x000002446BDCCC70> -
            key : device       value : cuda
            key : isFunction   value : False
            key : embList      value : tensor([-0.9335, -7.9620,  5.5738], device='cuda:0', requires_grad=True)
            key : embSize      value : 3
            key : value        value : a
    tensor(9.7638, device='cuda:0', grad_fn=<PowBackward0>)
    tensor([-0.0956, -0.8155,  0.5709], device='cuda:0') tensor([ 0.0956,  0.8155, -0.5709], device='cuda:0')
    [tensor(9.7638, device='cuda:0', grad_fn=<PowBackward0>)]
    '''
