from SDLib.baseclass.SDetection import SDetection
import numpy as np
import random
import scipy.sparse as sp


class FAP(SDetection):

    def __init__(self, conf, trainingSet=None, testSet=None, labels=None, fold='[1]'):
        super(FAP, self).__init__(conf, trainingSet, testSet, labels, fold)

    def readConfiguration(self):
        super(FAP, self).readConfiguration()
        # # s means the number of seedUser who be regarded as spammer in training
        self.s =int( self.config['seedUser'])
        # preserve the real spammer ID
        self.spammer = []
        for i in self.dao.user:
            if self.labels[i] == '1':
                self.spammer.append(self.dao.user[i])
        sThreshold = int(0.5 * len(self.spammer))
        if self.s > sThreshold :
            self.s = sThreshold
            print ('*** seedUser is more than a half of spammer, so it is set to', sThreshold, '***')

        # # predict top-k user as spammer
        self.k = int(self.config['topKSpam'])
        # 0.5 is the ratio of spammer to dataset, it can be changed according to different datasets
        kThreshold = int(0.5 * (len(self.dao.user) - self.s))
        if self.k > kThreshold:
            self.k = kThreshold
            print ('*** the number of top-K users is more than threshold value, so it is set to', kThreshold, '***')
    # product transition probability matrix self.TPUI and self.TPIU

    def __computeTProbability(self):
        # m--user count; n--item count
        m, n, tmp = self.dao.trainingSize()
        self.TPUI = [], [], []
        self.TPIU = [], [], []

        self.userUserIdDic = {}
        self.itemItemIdDic = {}
        self.reveseitemItemIdDic = {}
        tmpUser = list(self.dao.user.values())
        tmpUserId = list(self.dao.user.keys())
        tmpItem = list(self.dao.item.values())
        tmpItemId = list(self.dao.item.keys())
        # tmpUser = self.dao.user.values()
        # tmpUserId = self.dao.user.keys()
        # tmpItem = self.dao.item.values()
        # tmpItemId = self.dao.item.keys()
        for users in range(0, m):
            self.userUserIdDic[tmpUser[users]] = tmpUserId[users]
        for items in range(0, n):
            self.itemItemIdDic[tmpItem[items]] = tmpItemId[items]
            self.reveseitemItemIdDic[tmpItemId[items]] = tmpItem[items]
        for i in range(0, m):
            user = self.userUserIdDic[i]
            if user not in self.bipartiteGraphUI:
                continue
            for item in self.bipartiteGraphUI[user]:
                j = self.reveseitemItemIdDic[item]
                w = float(self.bipartiteGraphUI[user][item])
                # to avoid positive feedback and reliability problem,we should Polish the w
                otherItemW = 0
                otherUserW = 0
                for otherItem in self.bipartiteGraphUI[user]:
                    otherItemW += float(self.bipartiteGraphUI[user][otherItem])
                for otherUser in self.dao.trainingSet_i[item]:
                    otherUserW += float(self.bipartiteGraphUI[otherUser][item])
                # wPrime = w*1.0/(otherUserW * otherItemW)
                wPrime = w

                self.TPUI[0].append(wPrime / otherItemW)
                self.TPUI[1].append(i)
                self.TPUI[2].append(j)
                self.TPIU[0].append(wPrime / otherUserW)
                self.TPIU[1].append(j)
                self.TPIU[2].append(i)
            if i % 1000 == 0:
                 print ('progress: %d/%d' %(i,m))
        self.TPUI = sp.coo_matrix((self.TPUI[0], (self.TPUI[1], self.TPUI[2])), shape=(m, n), dtype=np.float32).tocsr()
        self.TPIU = sp.coo_matrix((self.TPIU[0], (self.TPIU[1], self.TPIU[2])), shape=(n, m), dtype=np.float32).tocsr()


    def initModel(self):
        # construction of the bipartite graph
        # print ("constructing bipartite graph...")
        self.bipartiteGraphUI = {}
        for user in self.dao.trainingSet_u:
            tmpUserItemDic = {}  # user-item-point
            for item in self.dao.trainingSet_u[user]:
                # tmpItemUserDic = {}#item-user-point
                recordValue = float(self.dao.trainingSet_u[user][item])
                w = 1 + abs((recordValue - self.dao.userMeans[user]) / self.dao.userMeans[user]) + abs(
                    (recordValue - self.dao.itemMeans[item]) / self.dao.itemMeans[item]) + abs(
                    (recordValue - self.dao.globalMean) / self.dao.globalMean)
                # tmpItemUserDic[user] = w
                tmpUserItemDic[item] = w
            # self.bipartiteGraphIU[item] = tmpItemUserDic
            self.bipartiteGraphUI[user] = tmpUserItemDic
        # we do the polish in computing the transition probability
        # print ("computing transition probability...")
        self.__computeTProbability()

    def isConvergence(self, PUser, PUserOld):
        if len(PUserOld) == 0:
            return True
        for i in range(0, len(PUser)):
            if abs(PUser[i] - PUserOld[i]) > 0.001:
                return True
        return False

    def buildModel(self):
        # -------init--------
        m, n, tmp = self.dao.trainingSize()
        PUser = np.zeros(m)
        PItem = np.zeros(n)
        self.testLabels = [0 for i in range(m)]
        self.predLabels = [0 for i in range(m)]

        # preserve seedUser Index
        self.seedUser = []
        randDict = {}
        for i in range(0, self.s):
            randNum = random.randint(0, len(self.spammer) - 1)
            while randNum in randDict:
                randNum = random.randint(0, len(self.spammer) - 1)
            randDict[randNum] = 0
            self.seedUser.append(int(self.spammer[randNum]))
            # print len(randDict), randDict
        for i in range(0, self.s * 3):
            randNum = random.randint(0, m - 1)
            while randNum in self.seedUser:
                randNum = random.randint(0, m - 1)
            self.seedUser.append(randNum)

        #initial user and item spam probability
        for j in range(0, m):
            if j in self.seedUser:
                #print type(j),j
                PUser[j] = 1
            else:
                PUser[j] = random.random()
        for tmp in range(0, n):
            PItem[tmp] = random.random()

        # -------iterator-------
        PUserOld = []
        iterator = 0
        while self.isConvergence(PUser, PUserOld):
        #while iterator < 100:
            for j in self.seedUser:
                PUser[j] = 1
            PUserOld = PUser
            PItem = self.TPIU @ PUser
            PUser = self.TPUI @ PItem
            iterator += 1
            # print (self.foldInfo,'iteration', iterator)

        PUserDict = {}
        userId = 0
        for i in PUser:
            PUserDict[userId] = i
            userId += 1
        for j in self.seedUser:
            del PUserDict[j]

        self.PSort = sorted(PUserDict.items(), key=lambda d: d[1], reverse=True)


    def predict(self):
        # predLabels
        # top-k user as spammer
        spamList = []
        sIndex = 0
        while sIndex < self.k:
            spam = self.PSort[sIndex][0]
            spamList.append(spam)
            self.predLabels[spam] = 1
            sIndex += 1

        return_label = {}

        # trueLabels
        for user in self.dao.trainingSet_u:
            userInd = self.dao.user[user]
            # print type(user), user, userInd
            self.testLabels[userInd] = int(self.labels[user])
            return_label[int(user)] = self.predLabels[userInd]

        # delete seedUser labels
        differ = 0
        for user in self.seedUser:
            user = int(user - differ)
            # print type(user)
            del self.predLabels[user]
            del self.testLabels[user]
            differ += 1

        return self.predLabels, return_label
