from array import array
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class KDD(object):
    def __init__(self):
        self.pathSetBool1=False
        self.pathSetBool2 = False
        self.pathSetBool3 = False
        self.transFunc = MinMaxScaler()

    def transformVals(self,a):
        size = a.shape
        a = a.reshape(-1, 1)
        a = self.transFunc.fit_transform(a)
        a = a.reshape(-1, size[1])
        return a

    def invTransformVals(self,a):
        size = a.shape
        a = a.reshape(-1, 1)
        a = self.transFunc.inverse_transform(a)
        a = a.reshape(-1, size[1])
        return a

    def rbin1(self):
        if self.pathSetBool1:
            f = open(self.dataPath1, 'rb')
            ar = array('h')
            ar.fromstring(f.read())
            finalar = list(ar)
            labels = finalar[::901]
            del finalar[::901]
            data = [finalar[i:i + 900] for i in range(0, len(finalar), 900)]
            data=np.array(data)
            data=self.transformVals(data)
            #newd = []
            #for b in data:
            #    newb = []
            #    for i in range(0, len(b), 30):
            #        newb.append(b[i:i + 30])
            #    newd.append(newb)
            #data = np.array(newd).astype(np.float32)
            data = np.array(data).astype(np.float32)
            labels=np.array(labels)
            return labels, data
        else:
            print("Set datapath with -setData- function")

    def rbin2(self):
        if self.pathSetBool2:
            f1 = open(self.dataPath1, 'rb')
            ar1 = array('h')
            ar1.fromstring(f1.read())
            finalar1 = list(ar1)
            labels1 = finalar1[::901]
            del finalar1[::901]
            data1 = [finalar1[i:i + 900] for i in range(0, len(finalar1), 900)]
            data1 = np.array(data1)

            f2 = open(self.dataPath2, 'rb')
            ar2 = array('h')
            ar2.fromstring(f2.read())
            finalar2 = list(ar2)
            labels2 = finalar2[::901]
            del finalar2[::901]
            data2 = [finalar2[i:i + 900] for i in range(0, len(finalar2), 900)]
            data2 = np.array(data2)
            newtotal = []
            for i in range(0, len(data1)):
                newcomb = np.concatenate((data1[i], data2[i]))
                newtotal.append(newcomb)
            newtotal = np.array(newtotal)
            data = self.transformVals(newtotal)
            data = np.array(data).astype(np.float32)
            if labels1==labels2:
                labels = np.array(labels1)
                return labels, data
            else:
                print('!!!!!!!!!!!!!!!!!!Error in labels !!!!!!!!!!!!!!!')

        else:
            print("Set datapath with -setData- function")

    def rbin3(self):
        if self.pathSetBool3:
            f1 = open(self.dataPath1, 'rb')
            ar1 = array('h')
            ar1.fromstring(f1.read())
            finalar1 = list(ar1)
            labels1 = finalar1[::901]
            del finalar1[::901]
            data1 = [finalar1[i:i + 900] for i in range(0, len(finalar1), 900)]
            data1 = np.array(data1)

            f2 = open(self.dataPath2, 'rb')
            ar2 = array('h')
            ar2.fromstring(f2.read())
            finalar2 = list(ar2)
            labels2 = finalar2[::901]
            del finalar2[::901]
            data2 = [finalar2[i:i + 900] for i in range(0, len(finalar2), 900)]
            data2 = np.array(data2)

            f3 = open(self.dataPath3, 'rb')
            ar3 = array('h')
            ar3.fromstring(f3.read())
            finalar3 = list(ar3)
            labels3 = finalar3[::901]
            del finalar3[::901]
            data3 = [finalar3[i:i + 900] for i in range(0, len(finalar3), 900)]
            data3 = np.array(data3)

            newtotal = []
            for i in range(0, len(data1)):
                newcomb = np.concatenate((data1[i], data2[i],data3[i]))
                newtotal.append(newcomb)

            newtotal = np.array(newtotal)
            data = self.transformVals(newtotal)
            data = np.array(data).astype(np.float32)
            if labels1==labels2==labels3:
                labels = np.array(labels1)
                return labels, data
            else:
                print('!!!!!!!!!!!!!!!!!!Error in labels !!!!!!!!!!!!!!!')

        else:
            print("Set datapath with -setData- function")


    def setData(self,*args):
        if len(args) == 2:
            device = args[0]
            feature= args[1]
            self.dataPath1='KDD/'+device+feature+'.bin'
            print('Data path set to: '+ self.dataPath1)
            self.pathSetBool1=True
            self.Labels , self.Data = self.rbin1()
        elif len(args) == 3:
            device1 = args[0]
            device2 = args[1]
            feature = args[2]
            self.dataPath1 = 'KDD/' + device1 + feature + '.bin'
            self.dataPath2 = 'KDD/' + device2 + feature + '.bin'
            print('Data path set to: ' + self.dataPath1 + 'and' + self.dataPath2)
            self.pathSetBool1 = True
            self.pathSetBool2 = True
            self.Labels, self.Data = self.rbin2()
        elif len(args) ==4:
            device1 = args[0]
            device2 = args[1]
            device3 = args[2]
            feature = args[3]
            self.dataPath1 = 'KDD/' + device1 + feature + '.bin'
            self.dataPath2 = 'KDD/' + device2 + feature + '.bin'
            self.dataPath3 = 'KDD/' + device3 + feature + '.bin'
            print('Data path set to: ' + self.dataPath1 + ',' + self.dataPath2 + 'and' + self.dataPath3)
            self.pathSetBool1 = True
            self.pathSetBool2 = True
            self.pathSetBool3 = True
            self.Labels, self.Data = self.rbin3()


    def next_batch(self,batchsize, data, labels):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:batchsize]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.array(data_shuffle),np.array(labels_shuffle)

    def getBatch(self,batchsize):
        D , L = self.next_batch(batchsize,self.Data,self.Labels)
        return D , L