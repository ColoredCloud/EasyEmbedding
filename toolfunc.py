import copy,torch

ShowWarning = {
    '__sub__':False,
    '__add__': True,
    'distanceInMultipleDimension': False,
    'nestmap': True,

}

def setWarning(bool):
    for k in ShowWarning:
        ShowWarning[k] = bool

def noWarning(func,*args,**kwargs):
    global ShowWarning
    _state = copy.copy(ShowWarning)
    setWarning(False)
    try:
        result = func(*args,**kwargs)
    except Exception as e:
        pass
    ShowWarning = _state
    return result

class Utools():
    def __init__(self):
        self.setWarning = setWarning
        self.showWarning = ShowWarning
    def colorWord(self,word,color='r',end='\n'):
        word = str(word)
        colorDict = {'r':91,'g':92,'y':93,'b':94,'m':95,'c':96}
        def toColor(int):
            return f'\033[{int}m'
        print(toColor(colorDict[color])+word+toColor(0),end=end)

    def raiseError(self,type,methodName,ctn=False):
        #print(methodName in ShowWarning and not ShowWarning[methodName])

        if methodName in ShowWarning and not ShowWarning[methodName]:
            return True


        self.colorWord(f'Error occurs in {methodName} : {type}'+(' -' if ctn else ' '))
        return False


Ut = Utools()