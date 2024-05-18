import torch
from tqdm import tqdm
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
            for param, grad in veclist:
                param -= self.lr * grad
                grad.zero_()


class Bundle():
    def __init__(self,tensorVec:TensorVec,lr=0):
        self.tensorVec = tensorVec
        self.optim = SpecificOptimizer(tensorVec, lr)
        self.associate = {} # {Bonded:distance}
        self.lr = lr

    def getDifference(self, other, weight, bias, distance=0): #distance = -1 : Bundles are indepenent
        if type(other) != Bundle:
            Ut.raiseError(f"Need Bundle object in parameter : other, got {type(other)}", sys._getframe().f_code.co_name)
            return
        #print(self.tensorVec.embList*weight + bias)d
        temp_embList = self.tensorVec.embList.clone()
        temp_embList = temp_embList * weight + bias
        temp_Vec = TensorVec(temp_embList,device=self.tensorVec.device)
        if distance == -1:
            d = (1/temp_Vec.getDistance(other.tensorVec))
        else:
            d = torch.abs(temp_Vec.getDistance(other.tensorVec)- distance)/Normalizer.std
        return d

    def set_lr(self, lr):
        self.lr = lr
        self.optim.set_lr(lr)

    def add(self,other,distance:int):
        if type(other) != Bundle:
            Ut.raiseError(f"Need Bundle object in parameter : other, got {type(other)}", sys._getframe().f_code.co_name)
            return
        self.associate[other] = distance

    def forward(self, Bundles:list, weight, bias, optim = False, lable = None):
        totalDiff = 0
        if self.lr == 0:
            Ut.raiseError("Need to set learning rate", sys._getframe().f_code.co_name)
            return
        for other in Bundles:
            if type(other) != Bundle:
                Ut.raiseError(f"Need Bundle object in parameter's value: Bondles, got {type(other)}",
                              sys._getframe().f_code.co_name)
                return
            if other == self:
                continue
            diff = self.getDifference(other, weight, bias, distance=self.associate[other] if other in self.associate else -1)
            totalDiff += diff
            if lable and lable == other.val():
                d1 = diff
        if optim == True:
            if lable is None:
                Ut.raiseError("Need lable to optimize, optim is forced to set to false", sys._getframe().f_code.co_name)
            else:
                totalDiff.backward()
                self.optim()


                b1 = [bd for bd in Bundles if bd.val() == lable][0]

                d1 = self.getDifference(b1, weight, bias, distance=self.associate[b1] if b1 in self.associate else -1)
                d1.backward()
                with torch.no_grad():
                    if weight.grad is not None:
                        weight.data -= self.lr * weight.grad
                        bias.data -= self.lr * bias.grad
                        weight.grad.zero_()
                        bias.grad.zero_()
                return totalDiff*d1

        return totalDiff

    def __str__(self):
        return str(self.tensorVec)

    def __repr__(self):
        return f'value : {self.val()}   Position : {str(self.tensorVec)}   Associated : {[aso.val() + f"  distance : {self.associate[aso]}"for aso in self.associate.keys()]}'

    def val(self):
        return self.tensorVec.value

class BundleList():
    def __init__(self):
        self.Bundles = []
        self.trainLoader=None
        self.device = 'cpu'
        self.setEmbSize(2)
        self.lr = None

    def __iter__(self):
        return self.Bundles.__iter__()

    def setEmbSize(self,embsize):
        if not isinstance(embsize,int) or embsize < 1:
            Ut.raiseError("Embedding size must be a positive integer", sys._getframe().f_code.co_name)
            return
        self.embsize = embsize
        self.weight = torch.randn(embsize,device=self.device,requires_grad=True)
        self.bias = torch.randn(embsize, device=self.device,requires_grad=True)

    def loadTrainSet(self,dataset):
        if any(not isinstance(s,str) for s in dataset):
            Ut.raiseError("dataset must be a list of strings", sys._getframe().f_code.co_name)
            return
        self.trainLoader = [s.split(' ') for s in dataset]

    def wordIsInBundles(self,word):
        if not isinstance(word,str):
            Ut.raiseError("word must be a string", sys._getframe().f_code.co_name)
            return
        return any(word == bundle.val() for bundle in self.Bundles)

    def getWordBundles(self,word):
        if not isinstance(word,str):
            Ut.raiseError("word must be a string", sys._getframe().f_code.co_name)
            return
        for bundle in self.Bundles:
            if word == bundle.val():
                return bundle
        Ut.raiseError("word not in bundles", sys._getframe().f_code.co_name)
    def InitializeTrainDataSet(self):
        if self.trainLoader is None:
            Ut.raiseError("No train set loaded", sys._getframe().f_code.co_name)
        for sentence in self.trainLoader:
            for wordai in range(len(sentence)):
                worda = sentence[wordai]
                for wordbi in range(len(sentence)):
                    wordb = sentence[wordbi]

                    if not self.wordIsInBundles(worda):
                        self.Bundles.append(Bundle(Vec(embnum=self.embsize, value=sentence[wordai], device=self.device), lr=self.lr))

                    if worda == wordb: continue

                    if not self.wordIsInBundles(wordb):
                        self.Bundles.append(Bundle(Vec(embnum=self.embsize, value=sentence[wordbi], device=self.device), lr=self.lr))

                    self.getWordBundles(worda).add(self.getWordBundles(wordb), distance=abs(wordai - wordbi))

    def set_lr(self, lr):
        for bundle in self.Bundles:
            bundle.set_lr(lr)
        self.lr = lr
    def train(self,epoch=1,lr=None):
        if lr is not None:
            self.set_lr(lr)
        elif self.lr is None:
            Ut.raiseError("No learning rate set", sys._getframe().f_code.co_name)
            return

        lossSum = 0
        for e in range(epoch):
            with tqdm(range(len(self.Bundles)), desc=f'epoch : {e} loss = {round(float(lossSum),Normalizer.round)} epoch : {e + 1} processingd') as t:

                lossSum = 0
                for bundle in self.Bundles:
                    loss = bundle.forward(self.Bundles,self.weight,self.bias,lable=bundle.val(),optim=True)
                    if loss is not None and not loss.isnan():
                        lossSum += loss
                    else:
                        Ut.raiseError(f"Fatal error : loss not a number", sys._getframe().f_code.co_name)
                        break
                    t.update(1)

    def __str__(self):
        output = {}
        for bundle in self.Bundles:
            output[bundle.val()] = str(bundle.tensorVec)
        return str(output)

    def repr(self, veryDetailed = False):
        if veryDetailed:
            for bundle in self.Bundles:
                print(repr(bundle))


        weight = str(list(funcs.nestmap(self.weight, lambda x: round(x, Normalizer.round))))
        if len(weight) > 35:
            weight = weight[:15] + ' ... ' + weight[-15:]
        Ut.colorWord(f'weight : {weight}', 'b')
        bias = str(list(funcs.nestmap(self.bias, lambda x: round(x, Normalizer.round))))
        if len(bias) > 35:
            bias = bias[:15] + ' ... ' + bias[-15:]

        Ut.colorWord(f'bias : {bias}', 'b')

if __name__ == '__main__':
    from Tensorvec import *
    model = Bundle(Vec([3, 2, 1],value='a'), lr=1)
    model1 = Bundle(Vec([1, 2, 3],value='b'), lr=1)
    model2 = Bundle(Vec([2, 2, 2], value='c'), lr=1)


    Bonds = [model,model1,model2]

    model.add(model1,distance=2)
    model.add(model2, distance=1)
    weight = torch.randn(3,requires_grad=True)
    bias = torch.randn(3,requires_grad=True)

    for epoch in range(10):
        print(model.forward(Bonds,weight=weight,bias=bias, optim=True, lable='b'))
        print(model.forward(Bonds, weight=weight, bias=bias, optim=True, lable='c'))
    print(model,weight,bias)
    '''
        Expect output
        
        tensor(0.0998, grad_fn=<MulBackward0>)
        tensor(0.0323, grad_fn=<MulBackward0>)
        tensor(6.0113e-05, grad_fn=<MulBackward0>)
        tensor(5.6061e-05, grad_fn=<MulBackward0>)
        tensor(0.0065, grad_fn=<MulBackward0>)
        tensor(0.0014, grad_fn=<MulBackward0>)
        tensor(0.0010, grad_fn=<MulBackward0>)
        tensor(0.0083, grad_fn=<MulBackward0>)
        tensor(0.0217, grad_fn=<MulBackward0>)
        tensor(0.0002, grad_fn=<MulBackward0>)
        tensor(0.0010, grad_fn=<MulBackward0>)
        tensor(0.0073, grad_fn=<MulBackward0>)
        tensor(0.0233, grad_fn=<MulBackward0>)
        tensor(0.0006, grad_fn=<MulBackward0>)
        tensor(0.0014, grad_fn=<MulBackward0>)
        tensor(0.0004, grad_fn=<MulBackward0>)
        tensor(0.0121, grad_fn=<MulBackward0>)
        tensor(3.0071e-05, grad_fn=<MulBackward0>)
        tensor(7.8916e-06, grad_fn=<MulBackward0>)
        tensor(0.0007, grad_fn=<MulBackward0>)
        [2.719, 1.824, 1.132] tensor([0.8105, 0.8966, 1.1809], requires_grad=True) tensor([-0.0036,  0.4519, -0.7566], requires_grad=True)

    '''