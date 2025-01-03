class DataAnalyser:

    _data = None

    def __init__(self, data):
        self.setData(data)
        
    def getCorrelMatrix(self):
        return self._data.corr()
    
    def getData(self):
        return self._data
    
    def setData(self, data):
        self._data = data