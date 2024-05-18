import torch
from toolfunc import Ut
from Normalize import Normalizer
from Tensorvec import *
from typing import Dict
class SpecificOptimizer():
    def __init__(self, model:TensorVec,lr=0.1):
        self.model = model
        self.lr = lr

    def set_lr(self, lr):
        self.lr = lr

    def __call__(self):
        veclist = zip(self.model.parameters(),self.model.grad())
        with torch.no_grad():
            for param,grad in veclist:
                param.data -= self.lr * grad
                grad.zero_()

class Bundle():
    def __init__(self,tensorVec:TensorVec,lr=0.1):
        self.tensorVec = tensorVec
        self.optim = SpecificOptimizer(tensorVec, lr)
        self.associate = {} # {Bonded:distance}

    def getLoss(self,other,distance=0,optim=False): #distance = -1 : Bundles are indepenent
        if type(other) != Bundle:
            Ut.raiseError(f"Need Bundle object in parameter : other, got {type(other)}", sys._getframe().f_code.co_name)
            return
        if distance == -1:
            l = (1/self.tensorVec.getDistance(other.tensorVec))
        else:
            l = torch.abs(self.tensorVec.getDistance(other.tensorVec)- distance)/Normalizer.std
        # print(l,distance)
        if optim:
            l.backward()
            self.optim()
            #other.optim()
        return l

    def set_lr(self, lr):
        self.optim.set_lr(lr)

    def add(self,other,distance:int):
        if type(other) != Bundle:
            Ut.raiseError(f"Need Bundle object in parameter : other, got {type(other)}", sys._getframe().f_code.co_name)
            return
        self.associate[other] = distance

    def forward(self,Bondles ,optim = True):
        loss = 0
        for other in Bondles:
            if type(other) != Bundle:
                Ut.raiseError(f"Need Bundle object in parameter's value: Bondles, got {type(other)}",
                              sys._getframe().f_code.co_name)
                return
            if other == self:
                continue
            loss += self.getLoss(other,distance=self.associate[other] if other in self.associate else -1,optim=optim)

        return loss

    def __str__(self):
        return str(self.tensorVec)

    def __repr__(self):
        return f'value : {self.val()}   Position : {str(self.tensorVec)}   Associated : {[aso.val() + f"  distance : {self.associate[aso]}"for aso in self.associate.keys()]}'

    def val(self):
        return self.tensorVec.value



if __name__ == '__main__':
    from Tensorvec import *
    model = Bundle(Vec([0, 0, 0],value='a'), lr=1)
    model1 = Bundle(Vec([1, 1, 1],value='b'), lr=1)
    Bonds = [model,model1]
    model.add(model1,0)
    model1.add(model, 0)
    print(model1.forward(Bonds,optim=False))
    for epoch in range(5):
        print(model.forward(Bonds))
    print(model1.forward(Bonds),end='\n\n')
    print(model)
    print(model1)
    '''
        Expect output
        
        tensor(0.1732, grad_fn=<AddBackward0>)
        tensor(0.1732, grad_fn=<AddBackward0>)
        tensor(0.1632, grad_fn=<AddBackward0>)
        tensor(0.1532, grad_fn=<AddBackward0>)
        tensor(0.1432, grad_fn=<AddBackward0>)
        tensor(0.1332, grad_fn=<AddBackward0>)
        tensor(0.1232, grad_fn=<AddBackward0>)
        
        [0.289, 0.289, 0.289]
        [0.654, 0.654, 0.654]

    '''