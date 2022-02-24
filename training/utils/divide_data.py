# -*- coding: utf-8 -*-
from random import Random
#from core.dataloader import DataLoader
from torch.utils.data import DataLoader
import numpy as np
from math import *
import logging
from scipy import stats
import numpy as np
from pyemd import emd
from collections import OrderedDict
import time
import pickle, random
from argParser import args

from torchvision import datasets



class Partition(object):
    """ Dataset partitioning helper 
        A thin wrapper around the torchvision.dataset object to
        pseudo-randomize access according to the partition
    """

    def __init__(self, data:datasets, index:list):
        self.data = data                # contains the dataset.Dataset
        self.index = index              # list of indices (corresponding to self.labels[]) from a partition

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    import torch
    # len(sizes) is the number of workers
    # sequential 1-> random 2->zipf 3-> identical
    def __init__(self, data:torch.utils.data.Datasets, numOfClass:int=0, seed=10, splitConfFile=None, isTest=False, dataMapFile=None):
        self.partitions = []
        self.rng = Random() # some random numer
        self.rng.seed(seed) # seed the random number
        self.data = data    # the datasets.HMDB51() datasets
        if isinstance(data, datasets.HMDB51):
            self.labels = [data.samples[vid_index][1] for vid_index in data.indices]    # data.samples = [(vid_index, class_index), ....]
                                                                                        # data.indices is those vid.index's where fold is matched
                                                                                        # That is data.indices specifies subset of data.samples
                                                                                        # self.labels has the class index's of vids specified by data.indices
            # self.index_vid_path = { index: vid_path for index, (vid_path, _) in enumerate(data.samples)}
        else :
            self.labels = self.data.targets                                     #[5,1,....] for MNIST i.e. class labels for the dataset
        self.is_trace = False
        self.dataMapFile = None
        self.args = args
        self.isTest = isTest                                                    # False by default

        np.random.seed(seed)                                                    # seed np.random

        stime = time.time()
        #logging.info("====Start to initiate DataPartitioner")

        self.targets = OrderedDict()                                            # set an OrderedDict for some labels
        self.indexToLabel = {}
        self.totalSamples = 0
        self.data_len = len(self.data)

        self.task = args.task                                                   # this is set to 'activity_recognition'
        self.skip_partition = True if self.labels[0] is -1 or args.skip_partition is True else False    # default becomes false but what does skip_partiton do ?

        if self.skip_partition:
            logging.info("====Warning: skip_partition is True")

        if self.skip_partition:
            pass

        elif splitConfFile is None:
            # categarize the samples
            for index, label in enumerate(self.labels):                         # Create an ordered dict of the form {'class_index':[self.labels index...]}
                if label not in self.targets:
                    self.targets[label] = []
                self.targets[label].append(index)                               # in case of hmdb51, the list contains index os elements in self.lables. 
                self.indexToLabel[index] = label                                # self.labels contains the index of the actual data in self.data.samples

            self.totalSamples += len(self.data)
        else:
            # each row denotes the number of samples in this class
            with open(splitConfFile, 'r') as fin:
                labelSamples = [int(x.strip()) for x in fin.readlines()]

            # categarize the samples
            baseIndex = 0
            for label, _samples in enumerate(labelSamples):
                for k in range(_samples):
                    self.indexToLabel[baseIndex + k] = label
                self.targets[label] = [baseIndex + k for k in range(_samples)]
                self.totalSamples += _samples
                baseIndex += _samples

        if dataMapFile is not None:
            self.dataMapFile = dataMapFile
            self.is_trace = True

        self.numOfLabels = max(len(self.targets.keys()), numOfClass)
        self.workerDistance = []
        self.classPerWorker = None                                              # a numpy array of dim # of workers X # of classes specifying how many vids per class

        logging.info("====Initiating DataPartitioner takes {} s\n".format(time.time() - stime))

    def getTargets(self):
        tempTarget = self.targets.copy()

        for key in tempTarget:
            self.rng.shuffle(tempTarget[key])

        return tempTarget

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    # Calculates JSD between pairs of distribution
    def js_distance(self, x, y):
        m = (x + y)/2
        js = 0.5 * stats.entropy(x, m) + 0.5 * stats.entropy(y, m)
        return js

    # Caculates Jensen-Shannon Divergence for each worker
    def get_JSD(self, dataDistr, tempClassPerWorker, sizes):
        for worker in range(len(sizes)):
            # tempDataSize = sum(tempClassPerWorker[worker])
            # if tempDataSize == 0:
            #     continue
            # tempDistr =np.array([c / float(tempDataSize) for c in tempClassPerWorker[worker]])
            self.workerDistance.append(0)#self.js_distance(dataDistr, tempDistr))

    # Generates a distance matrix for EMD
    def generate_distance_matrix(self, size):
        return np.logical_xor(1, np.identity(size)) * 1.0

    # Caculates Earth Mover's Distance for each worker
    def get_EMD(self, dataDistr, tempClassPerWorker, sizes):
        dist_matrix = self.generate_distance_matrix_v2(len(dataDistr))
        for worker in range(len(sizes)):
            tempDataSize = sum(tempClassPerWorker[worker])
            if tempDataSize == 0:
                continue
            tempDistr =np.array([c / float(tempDataSize) for c in tempClassPerWorker[worker]])
            self.workerDistance.append(emd(dataDistr, tempDistr, dist_matrix))

    def loadFilterInfo(self):
        # load data-to-client mapping
        indicesToRm = []

        try:
            dataToClient = OrderedDict()

            with open(self.args.data_mapfile, 'rb') as db:
                dataToClient = pickle.load(db)

            clientNumSamples = {}
            sampleIdToClient = []

            # data share the same index with labels
            for index, _sample in enumerate(self.data.data):
                sample = _sample.split('__')[0]
                clientId = dataToClient[sample]

                if clientId not in clientNumSamples:
                    clientNumSamples[clientId] = 0

                clientNumSamples[clientId] += 1
                sampleIdToClient.append(clientId)

            for index, clientId in enumerate(sampleIdToClient):
                if clientNumSamples[clientId] < self.args.filter_less:
                    indicesToRm.append(index)

        except Exception as e:
            logging.info("====Failed to generate indicesToRm, because of {}".format(e))
            #pass

        return indicesToRm

    def loadFilterInfoNLP(self):
        indices = []
        base = 0

        for idx, sample in enumerate(self.data.slice_index):
            if sample < args.filter_less:
                indices = indices + [base+i for i in range(sample)]
            base += sample

        return indices

    def loadFilterInfoBase(self):
        indices = []

        try:
            for client in self.data.client_mapping:         # most likely self.data.client_mapping is None
                if len(self.data.client_mapping[client]) < args.filter_less or len(self.data.client_mapping[client]) > args.filter_more:
                    indices += self.data.client_mapping[client]

                    # remove the metadata
                    for idx in self.data.client_mapping[client]:
                        self.data[idx] = None

        except Exception as e:
            pass

        return indices                                      # returns an empty list cuz, Dataset.client_mapping isn't existing

    def partitionTraceCV(self, dataToClient):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = self.numOfLabels

        # data share the same index with labels
        for index, sample in enumerate(self.data.data):
            sample = sample.split('__')[0]
            clientId = dataToClient[sample]
            labelId = self.labels[index]

            if clientId not in clientToData:
                clientToData[clientId] = []
                clientNumSamples[clientId] = [0] * numOfLabels

            clientToData[clientId].append(index)
            clientNumSamples[clientId][labelId] += 1

        numOfClients = len(clientToData.keys())
        self.classPerWorker = np.zeros([numOfClients, numOfLabels])

        for clientId in range(numOfClients):
            self.classPerWorker[clientId] = clientNumSamples[clientId]
            self.rng.shuffle(clientToData[clientId])
            self.partitions.append(clientToData[clientId])

        overallNumSamples = np.asarray(self.classPerWorker.sum(axis=0)).reshape(-1)
        totalNumOfSamples = self.classPerWorker.sum()

        self.get_JSD(overallNumSamples/float(totalNumOfSamples), self.classPerWorker, [0] * numOfClients)

    def partitionTraceSpeech(self, dataToClient):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = 35

        # data share the same index with labels

        for index, sample in enumerate(self.data.data):
            clientId = dataToClient[sample]
            labelId = self.labels[index]

            if clientId not in clientToData:
                clientToData[clientId] = []
                clientNumSamples[clientId] = [0] * numOfLabels

            clientToData[clientId].append(index)
            clientNumSamples[clientId][labelId] += 1

        numOfClients = len(clientToData.keys())
        self.classPerWorker = np.zeros([numOfClients, numOfLabels])

        for clientId in range(numOfClients):
            #logging.info(clientId)
            self.classPerWorker[clientId] = clientNumSamples[clientId]
            self.rng.shuffle(clientToData[clientId])
            self.partitions.append(clientToData[clientId])

        overallNumSamples = np.asarray(self.classPerWorker.sum(axis=0)).reshape(-1)
        totalNumOfSamples = self.classPerWorker.sum()

        self.get_JSD(overallNumSamples/float(totalNumOfSamples), self.classPerWorker, [0] * numOfClients)

    def partitionTraceNLP(self):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = 1
        base = 0
        numOfClients = 0

        numOfLabels = self.args.num_class
        for index, cId in enumerate(self.data.dict.keys()):
            clientId = cId
            labelId = self.data.targets[index]

            if clientId not in clientToData:
                clientToData[clientId] = []
                clientNumSamples[clientId] = [0] * numOfLabels
            clientToData[clientId].append(index)

        numOfClients = len(self.clientToData)

    def partitionTraceBase(self):
        clientToData = {}
        clientNumSamples = {}
        numOfLabels = self.args.num_class

        clientToData = self.data.client_mapping
        for clientId in clientToData:
            clientNumSamples[clientId] = [1] * numOfLabels

        numOfClients = len(clientToData)
        self.classPerWorker = np.zeros([numOfClients+1, numOfLabels])

        for clientId in range(numOfClients):
            self.classPerWorker[clientId] = clientNumSamples[clientId]
            self.rng.shuffle(clientToData[clientId])
            self.partitions.append(clientToData[clientId])

            # if len(clientToData[clientId]) < args.filter_less or len(clientToData[clientId]) > args.filter_more:
            #     # mask the raw data
            #     for idx in clientToData[clientId]:
            #         self.data[idx] = None

        overallNumSamples = np.asarray(self.classPerWorker.sum(axis=0)).reshape(-1)
        totalNumOfSamples = self.classPerWorker.sum()

        self.get_JSD(overallNumSamples/float(totalNumOfSamples), self.classPerWorker, [0] * numOfClients)

    def partitionDataByDefault(self, sizes, sequential, ratioOfClassWorker, filter_class, _args):
        if self.is_trace and not self.args.enforce_random:          # unless dataMapFile give, is_trace = False
            # use the real trace, thus no need to partition
            if self.task == 'speech' or self.task == 'cv':
                dataToClient = OrderedDict()

                with open(self.dataMapFile, 'rb') as db:
                    dataToClient = pickle.load(db)

                if self.task == 'speech':
                    self.partitionTraceSpeech(dataToClient=dataToClient)
                else:
                    self.partitionTraceCV(dataToClient=dataToClient)
            else:
                self.partitionTraceBase()
        else:
            self.partitionData(sizes=sizes, sequential=sequential,
                               ratioOfClassWorker=ratioOfClassWorker,
                               filter_class=filter_class, args=_args)

    def partitionData(self, sizes=None, sequential=0, ratioOfClassWorker=None, filter_class=0, args = None):
        """
        creates a partition matrix that basically shows how many vids per class per worker
        also modifies -> self.partitions(contains the vid id for each partition), 
        """
        targets = self.getTargets()                     # returns a shuffled self.targets OrderedDict - contains indices of self.labels for a class_index
        numOfLabels = self.getNumOfLabels()             # will most probably get 51 #TODO
        data_len = self.getDataLen()                    # returns the length of the subset of the  dataset i.e. vids specified in annotated dir and matched with fold

        usedSamples = 100000

        keyDir = {key:int(key) for i, key in enumerate(targets.keys())}         # a dict of the sort {class_index:class_index} for some weird reason :/
        keyLength = [0] * numOfLabels                                           # [0,0,0,0,.......,0] 51 labels and 51 zeros

        if not self.skip_partition:                                             # skip_partition is False
            for key in keyDir.keys():
                keyLength[keyDir[key]] = len(targets[key])                      # find out how many samples per class are in the dataset := keyLength[class_label] - # of samples

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])                # create a matrix of 4 X 51

        # random partition
        if sequential == 0:                                                     # this is our case for all clients
            logging.info("========= Start of Random Partition =========\n")

            # may need to filter ...
            indicesToRm = set()                                                 # This is the indices to remove
            indexes = None                                                      # The indexes of the "videos" (actually labels[] list) that will remain
            if self.args.filter_less != 0 and self.isTest is False:             # filter_less specifies the min number of trg samples of participating clients
                if self.task == 'cv':
                    indicesToRm = set(self.loadFilterInfo())
                else:
                    indicesToRm = set(self.loadFilterInfoBase())                # its still an empty set

                indexes = [x for x in range(0, data_len) if x not in indicesToRm]   # this becomes a list [0, 1, 2, ...., length of the (a)dataset - 1]
                # we need to remove those with less than certain number of samples
                logging.info("====Try to remove clients w/ less than {} samples, and remove {} samples".format(self.args.filter_less, len(indicesToRm)))
            else:
                indexes = [x for x in range(data_len)]

            self.rng.shuffle(indexes)                                           # why am i shuffling the index list ?
            realDataLen = len(indexes)                                          # still is length og (a) dataset - 1


            # create patitions which are lists of the lables[] indices to be used in a particular partition.
            # p1 = [2,14,35,1] means that p1 has the vids corresponding to labels[2,14,35,1]
            # henceforth alias labels[] with vids

            for ratio in sizes:                                                 # [1.0, 0.5, 0.33, 0.25]
                part_len = int(ratio * realDataLen)                             # number of elements in a partition
                self.partitions.append(indexes[0:part_len])                     # self.partitions.append a slice of part_len size of the shuffled list
                                                                                # Each partition := [the vid indexes to be used]  
                indexes = indexes[part_len:]                                    # Now remove the vid indices that have already been used in this partition

            if not self.skip_partition:                                         # skip partition is False by default
                for id, partition in enumerate(self.partitions):
                    for index in partition:
                        tempClassPerWorker[id][self.indexToLabel[index]] += 1   # indexToLabel[i] is basically labels[i]
                                                                                # so basically, the matrix will show how many vids in each class for each worker
        else:
            logging.info('========= Start of Class/Worker =========\n')

            if ratioOfClassWorker is None:
                # random distribution
                if sequential == 1:
                    ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels)
                # zipf distribution
                elif sequential == 2:
                    ratioOfClassWorker = np.random.zipf(args['param'], [len(sizes), numOfLabels])
                    logging.info("==== Load Zipf Distribution ====\n {} \n".format(repr(ratioOfClassWorker)))
                    ratioOfClassWorker = ratioOfClassWorker.astype(np.float32)
                else:
                    ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)

            if filter_class > 0:
                for w in range(len(sizes)):
                    # randomly filter classes by forcing zero samples
                    wrandom = self.rng.sample(range(numOfLabels), filter_class)
                    for wr in wrandom:
                        ratioOfClassWorker[w][wr] = 0.001

            # normalize the ratios
            if sequential == 1 or sequential == 3:
                sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)
                for worker in range(len(sizes)):
                    ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :]/float(sumRatiosPerClass[worker])

                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = min(floor(usedSamples * ratioOfClassWorker[worker][keyDir[c]]), keyLength[keyDir[c]])
                        self.rng.shuffle(targets[c])
                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength

                    self.rng.shuffle(self.partitions[-1])
            elif sequential == 2:
                sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=0)
                for c in targets.keys():
                    ratioOfClassWorker[:, keyDir[c]] = ratioOfClassWorker[:, keyDir[c]]/float(sumRatiosPerClass[keyDir[c]])

                # split the classes
                for worker in range(len(sizes)):
                    self.partitions.append([])
                    # enumerate the ratio of classes it should take
                    for c in list(targets.keys()):
                        takeLength = min(int(math.ceil(keyLength[keyDir[c]] * ratioOfClassWorker[worker][keyDir[c]])), len(targets[c]))
                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength
                        targets[c] = targets[c][takeLength:]

                    self.rng.shuffle(self.partitions[-1])

            elif sequential == 4:
                # load data from given config file
                clientGivenSamples = {}
                with open(args['clientSampleConf'], 'r') as fin:
                    for clientId, line in enumerate(fin.readlines()):
                        clientGivenSamples[clientId] = [int(x) for x in line.strip().split()]

                # split the data
                for clientId in range(len(clientGivenSamples.keys())):
                    self.partitions.append([])

                    for c in list(targets.keys()):
                        takeLength = clientGivenSamples[clientId][c]
                        if clientGivenSamples[clientId][c] > targets[c]:
                            logging.info("========== Failed to allocate {} samples for class {} to client {}, actual quota is {}"\
                                .format(clientGivenSamples[clientId][c], c, clientId, targets[c]))
                            takeLength = targets[c]

                        self.partitions[-1] += targets[c][0:takeLength]
                        tempClassPerWorker[worker][keyDir[c]] += takeLength
                        targets[c] = targets[c][takeLength:]

                self.rng.shuffle(self.partitions[-1])

        # concatenate ClassPerWorker
        if self.classPerWorker is None:                                         # yes for hmdb default case
            self.classPerWorker = tempClassPerWorker                            # assign the matrix
        else:
            self.classPerWorker = np.concatenate((self.classPerWorker, tempClassPerWorker), axis=0)

        # Calculates statistical distances
        totalDataSize = max(sum(keyLength), 1)                                  # sum of total number of videos of each class
        # Overall data distribution
        dataDistr = np.array([key / float(totalDataSize) for key in keyLength])
        self.get_JSD(dataDistr, tempClassPerWorker, sizes)                      # calculate the JSD #TODO have skipped for now
                                                                                # get_JSD hasa side_effect on self.workerdistance

        logging.info("Raw class per worker is : " + repr(tempClassPerWorker) + '\n')
        logging.info('========= End of Class/Worker =========\n')

    def log_selection(self):

        # totalLabels = [0 for i in range(len(self.classPerWorker[0]))]
        # logging.info("====Total # of workers is :{}, w/ {} labels, {}, {}".format(len(self.classPerWorker), len(self.classPerWorker[0]), len(self.partitions), len(self.workerDistance)))

        # for index, row in enumerate(self.classPerWorker):
        #     rowStr = ''
        #     numSamples = 0
        #     for i, label in enumerate(self.classPerWorker[index]):
        #         rowStr += '\t'+str(int(label))
        #         totalLabels[i] += label
        #         numSamples += label

        #     logging.info(str(index) + ':\t' + rowStr + '\n' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index]))+ '\nDistance: ' + str(self.workerDistance[index])+ '\n')
        #     logging.info("=====================================\n")

        # logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        # logging.info("=====================================\n")

        # remove unused variables

        self.classPerWorker = None
        self.numOfLabels = None
        pass

    def use(self, partition, istest, is_rank, fractional):
        _partition = partition                          # this is 0 for the first run :/
        resultIndex = []

        if is_rank == -1:                               # first run is_rank = -1
            resultIndex = self.partitions[_partition]   # select the last partition's vid index list
        else:
            for i in range(len(self.partitions)):
                if i % self.args.total_worker == is_rank:
                    resultIndex += self.partitions[i]

        exeuteLength = -1 if istest == False or fractional == False else int(len(resultIndex) * args.test_ratio)    # first run --> istest = False

        resultIndex = resultIndex[:exeuteLength]        # drop the last index from the list
        self.rng.shuffle(resultIndex)                   # shuffle the list

        #logging.info("====Data length for client {} is {}".format(partition, len(resultIndex)))
        return Partition(self.data, resultIndex)        # self.data = dataset.HMDB51(), resultIndex = shuffled vid index (corresponding to label)
        # think of this as returning a partition sized view into the dataset

    def getDistance(self):
        return self.workerDistance

    def getSize(self):
        # return the size of samples
        return [len(partition) for partition in self.partitions]

def partition_dataset(partitioner, workers, partitionRatio=[], sequential=0, ratioOfClassWorker=None, filter_class=0, arg={'param': 1.95}):
    """ Partitioning Data """
    stime = time.time()
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]   # partition_sizes = [1.0, 0.5, 0.33, 0.25]

    if len(partitionRatio) > 0:                                         # given as an empty list
        partition_sizes = partitionRatio

    partitioner.partitionDataByDefault(sizes=partition_sizes, sequential=sequential, ratioOfClassWorker=ratioOfClassWorker,filter_class=filter_class, _args=arg)
    # call is essentially - partitionDataByDefault([1,0, 0,5, 0.33, 0.25 ], 0, None, 0, {'param':5.0,...})

    #logging.info("====Partitioning data takes {} s\n".format(time.time() - stime()))

def select_dataset(rank: int, partition: DataPartitioner, batch_size: int, isTest=False, is_rank=0, fractional=True, collate_fn=None) -> DataLoader:

    partition = partition.use(rank - 1, isTest, is_rank-1, fractional)  # returns a Partition object
    timeOut = 0 if isTest else 60
    numOfThreads = args.num_loaders #int(min(args.num_loaders, len(partition)/(batch_size+1)))      # default value is 2
    dropLast = False if isTest else True

    if collate_fn is None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=numOfThreads, drop_last=dropLast, timeout=timeOut)#, worker_init_fn=np.random.seed(12))
    else:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=numOfThreads, drop_last=dropLast, timeout=timeOut, collate_fn=collate_fn)#, worker_init_fn=np.random.seed(12))

