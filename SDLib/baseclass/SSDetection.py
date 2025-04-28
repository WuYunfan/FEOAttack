from SDetection import SDetection
from SDLib.data.social import SocialDAO
class SSDetection(SDetection):

    def __init__(self,conf,trainingSet=None,testSet=None,labels=None,relation=list(),fold='[1]'):
        super(SSDetection, self).__init__(conf,trainingSet,testSet,labels,fold)
        self.sao = SocialDAO(self.config, relation)  # social relations access control
