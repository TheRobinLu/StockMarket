import numpy


class loader():
    period = 120
    filename = 'GSPC.txt'
    rawData = []
    trainData = []
    target = []
    closeData = []

    # def __init__(self):
        # self.filename = 'GSPC.txt'

    def load(self):
        f = open(self.filename, "r")
        if f.mode == 'r':
            content = f.read()
        data = numpy.fromstring(content, dtype=float, sep='\t')
        data = numpy.reshape(data, (-1, 5))
        self.rawData = data
        return data

    def genTrainData(self):

        for i in range(len(self.rawData)):
            if i >=self.period and i < len(self.rawData) - 1:
                self.trainData.append(self.rawData[i - self.period: i ] )

                high = self.rawData[i + 1, 1] - 0.2 * (self.rawData[i + 1, 1] - self.rawData[i + 1, 2])
                low = self.rawData[i + 1, 2] + 0.2 * (self.rawData[i + 1, 1] - self.rawData[i + 1, 2])
                middle = (self.rawData[i + 1,0] + self.rawData[i + 1,1] +self.rawData[i + 1,2] + 2 * self.rawData[i + 1,3])/5
                self.target.append([high, low, middle])

        self.target = numpy.asarray(self.target)
        self.trainData = numpy.asarray(self.trainData)

    def normalize(self):
        for i in range(len(self.trainData)):
            close = self.trainData[i, self.period - 1, 3]
            self.closeData.append(close)
            avgVol = numpy.average(self.rawData[:,4])
            for j in range(self.period):
                self.trainData[i,j,4] = self.trainData[i,j,4]/avgVol
                for k in range(4):
                    self.trainData[i, j, k] = self.trainData[i, j, k]/close

            for m in range(3):
                self.target[i,m] = self.target[i, m]/close









# class trade():
#
# myloader = loader()
#
# a = myloader.load()
#
# myloader.genTrainData()
# a1 = 2