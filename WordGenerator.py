import torch

from Optimizer import BundleList
from Tensorvec import funcs
from toolfunc import Ut
from Normalize import Normalizer

import math,sys
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
            lossSum +=generated_bundle.getDifference(bundle, weight=bundlelist.weight,bias=bundlelist.bias, distance=len(generated_text_bundle) - generated_bundle_index)

        lossDenominator += lossSum
        Losss[val] = lossSum

    if len(Losss) == 1:
        return list(Losss.keys())[0]

    weights = list(map(lambda x:1-x/lossDenominator,Losss.values()))
    key = random.choices(list(Losss.keys()), weights=weights, k=1)[0]
    return key

if __name__ == '__main__':

    Normalizer.set('round',2)
    Normalizer.normal(0,5)

    trainTest = ['How are you?','Fine thank you, and you?','Nice to meet you!','Who are you?']
    Bundles = BundleList()
    Bundles.setEmbSize(10)
    Bundles.loadTrainSet(trainTest)
    Bundles.InitializeTrainDataSet()


    #Bundles.repr()

    Bundles.train(epoch=20,lr=1)
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
