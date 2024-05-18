import torch

from Optimizer import Bundle
from Tensorvec import Vec
from toolfunc import Ut
from Normalize import Normalizer
import random,time,math,sys
from tqdm import tqdm
import os

def SimpleWordGenerator(generated_text_bundle, bundlelist):

    Losss = {}
    lossDenominator = 0

    for bundle in bundlelist:
        lossSum = 0
        val = bundle.val()
        if any(generated_bundle.val() == val for generated_bundle in generated_text_bundle):
            continue
        for generated_bundle_index in range(len(generated_text_bundle)):
            generated_bundle = generated_text_bundle[generated_bundle_index]
            lossSum +=generated_bundle.getLoss(bundle, optim=False, distance=len(generated_text_bundle) - generated_bundle_index)

        lossDenominator += lossSum
        Losss[val] = lossSum

    if len(Losss) == 1:
        return list(Losss.keys())[0]

    weights = list(map(lambda x:1-x/lossDenominator,Losss.values()))
    key = random.choices(list(Losss.keys()), weights=weights, k=1)[0]
    return key



class BundleList():
    def __init__(self):
        self.Bundles = []
        self.trainLoader=None
        self.device = 'cpu'
        self.embsize = 2
        self.lr = None

    def __iter__(self):
        return self.Bundles.__iter__()
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
                    loss = bundle.forward(self.Bundles)
                    if loss is not None and not loss.isnan():
                        lossSum += loss
                    else:
                        Ut.raiseError(f"Fatal error : loss not a number", sys._getframe().f_code.co_name)
                    t.update(1)

    def __str__(self):
        output = {}
        for bundle in self.Bundles:
            output[bundle.val()] = str(bundle.tensorVec)
        return str(output)

    def repr(self):
        for bundle in self.Bundles:
            print(repr(bundle))
if __name__ == '__main__':

    Normalizer.set('round',2)
    Normalizer.normal(0,5)

    trainTest = ['How are you?','Fine thank you, and you?','Nice to meet you!','Who are you?']
    #trainTest = ['a','b','c']
    Bundles = BundleList()
    Bundles.loadTrainSet(trainTest)
    Bundles.InitializeTrainDataSet()


    #Bundles.repr()

    Bundles.train(epoch=20,lr=5)
    os.system('cls')
    Bundles.repr()

    while True:
        wordsIn = [Bundles.getWordBundles(input('\nStart with : '))]
        length=input('length : ')

        print(wordsIn[-1].val(), end=' ')
        for i in range(int(length)):
            n = SimpleWordGenerator(wordsIn, bundlelist=Bundles)
            wordsIn.append(Bundles.getWordBundles(n))
            print(n,end=' ')
