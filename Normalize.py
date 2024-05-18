class Normal(object):
    def normal(self,mean,std):
        self.mean = mean
        self.std = std
        self.round = 3
        return self
    def set(self,name,value):
        self.__dict__[name] = value
    def __repr__(self):
        for i in self.__dict__.items():
            print(i)
        return super().__repr__()

Normalizer = Normal().normal(0,10)