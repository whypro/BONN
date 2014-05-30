# -*- coding: utf-8 -*-
from __future__ import division
import os
import pywt
import numpy as np
import zipfile
import random


class Segment(object):
    def __init__(self):
        self.classification = None
        self.epochs = []
        self.records = []

    # def __repr__(self):
    #     string = '\nclassification: ' + self.classification
    #     string += '\nrecords: \n' + '\n'.join([str(point) for point in self.records])
    #     string += '\nepochs: \n' + '\n'.join([str(epoch) for epoch in self.epochs])
    #     return string

class Epoch(object):
    def __init__(self):
        self.svm_classification = None
        self.points = []
        self.dwt_results = []
        self.features = []


class Processor(object):
    """读取 A.zip, B.zip, C.zip 后进行小波变换，特征提取后保存至 A.txt, B.txt, C.txt。"""
    CLASS_MAP = {'A': '0', 'D': '0', 'E': '1'}

    def __init__(self, debug=False):
        self.debug = debug
        self.segments = []

    def read_from_zipfile(self, filename):
        z = zipfile.ZipFile(filename, 'r')
        for fname in z.namelist():
            f = z.open(fname, 'r')
            segment = Segment()
            segment.classification = self.CLASS_MAP[filename[0]]
            for line in f:
                point = line.strip()
                if point:
                    segment.records.append(int(point))
            self.segments.append(segment)
            f.close()
        if self.debug:
            print '%d points has been read.' % sum(len(segment.records) for segment in self.segments)
            print '%d segment has been read.' % len(self.segments)
            # print self.records

    def calc_epoch(self, point_num=256, overlap=1/2):
        # segments -> segment(epochs) -> epoch(points)
        for i, segment in enumerate(self.segments):
            epoch_num = int((len(segment.records)-point_num*overlap) / (point_num*(1-overlap)))     # 默认条件下为 31
            for epoch_i in range(1, epoch_num+1):
                begin = int(point_num*overlap*(epoch_i-1))
                end = int(point_num*overlap*(epoch_i-1)+point_num)
                # print epoch_i, ',', begin, ':', end
                epoch = Epoch()
                epoch.points = segment.records[begin:end]
                #print len(epoch.points)
                segment.epochs.append(epoch)
                # print i, len(segment.epochs)

        # print len(self.overlapped_segments)
        if self.debug:
            print 'segments length = %d, segment.epochs length = %d, epoch.points length = %d.' % (
                len(self.segments), 
                len(self.segments[0].epochs), 
                len(self.segments[0].epochs[0].points)
            )

    def discrete_wavelet_transform(self):
        # segments -> segment(epochs) -> results
        for segment in self.segments:
            for epoch in segment.epochs:
                epoch.dwt_results = pywt.wavedec(epoch.points, 'db4', level=3)

        if self.debug:
            print 'segments length = %d, segment.epochs length = %d, epoch.dwt_results length = %d.' % (
                len(self.segments), 
                len(self.segments[0].epochs), 
                len(self.segments[0].epochs[0].dwt_results)
            )

    def extract_features(self):
        # segments -> epochs -> features
        for segment in self.segments:
            for epoch in segment.epochs:
                for i in epoch.dwt_results:
                    average = self.calc_average(i)
                    variance = self.calc_diff_abs(i)
                    epoch.features.append(average)
                    epoch.features.append(variance)

        if self.debug:
            print 'segments length = %d, segment.epochs length = %d, epoch.features length = %d.' % (
                len(self.segments), 
                len(self.segments[0].epochs), 
                len(self.segments[0].epochs[0].features)
            )

    @staticmethod
    def calc_average(data):
        return sum(data)/len(data)

    @staticmethod
    def calc_variance(data, average=None):
        if not average:
            average = sum(data)/len(data)
        diff = []
        for point in data:
            diff.append((point-average)**2)
        return sum(diff)/len(diff)

    @staticmethod
    def calc_diff_abs(data, average=None):
        if not average:
            average = sum(data)/len(data)
        diff = []
        for point in data:
            diff.append(abs(point-average))
        return sum(diff)/len(diff)

    def shuffle(self):
        random.shuffle(self.segments)
        if self.debug:
            print '%d segments shuffled.' % len(self.segments)

    def __convert_to_libsvm_data(self, segment):
        """
        1:<cA3 Average> 2:<cA3 Variance> 3:<cD3 Average> 4:<cD3 Variance> 5:<cD2 Average> 6:<cD2 Variance> 7:<cD1 Average> 8:<cD1 Variance>
        """
        data = ''
        for epoch in segment.epochs:
            row = [segment.classification]
            for i, feature in enumerate(epoch.features, start=1):
                row.append('%d:%f' % (i, feature))
            data += ' '.join(row)
            data += '\n'
        return data

    def __convert_to_separated_data(self, segment):
        label_data = ''
        features_data = ''
        for epoch in segment.epochs:
            row = []
            for feature in epoch.features:
                row.append('%f' % feature)
            features_data += ' '.join(row)
            features_data += '\n'
            label_data += segment.classification + '\n'
        return label_data, features_data

    def save_train_and_test_data(self, training_filename='data', predicting_filename='data.test'):
        training_segments = self.segments[:2*len(self.segments)//3]
        predicting_segments = self.segments[2*len(self.segments)//3:]
        
        training_file = open(training_filename, 'w')
        for segment in training_segments:
            data = self.__convert_to_libsvm_data(segment)
            training_file.write(data)
        training_file.close()

        predicting_file = open(predicting_filename, 'w')
        for segment in predicting_segments:
            data = self.__convert_to_libsvm_data(segment)
            predicting_file.write(data) 
        predicting_file.close()

    def go(self):
        missions = {
            'A/Z.zip': 'A/Z.txt', 
            'D/F.zip': 'D/F.txt', 
            'E/S.zip': 'E/S.txt',
        }
        for i, o in missions.items():
            self.read_from_zipfile(i)
        
        self.calc_epoch()
        self.discrete_wavelet_transform()
        self.extract_features()
        self.shuffle()
        self.save_train_and_test_data()

        # self.reset()
        # self.merge(missions.values(), 'data')


class Evaluator(object):
    def __init__(self, testing_filename='data.test', predicting_filename='data.out'):
        self.segments = []
        self.testing_filename = testing_filename
        self.predicting_filename = predicting_filename

    def read_true_values(self, segment_num = 100, epoch_num=31):
        f = open(self.testing_filename, 'r')
        for i in range(0, segment_num):
            segment = Segment()
            for j in range(0, epoch_num):
                epoch = Epoch()
                segment.epochs.append(epoch)
                classification = f.readline().strip().split(' ')[0]
                # print classification,
            # print '\n'
            segment.classification = classification
            # print segment.classification
            self.segments.append(segment)

        f.close()
        print '%d lines has been read.' % (i+1)
        print '%d segments has been read.' % len(self.segments)

    def read_predicted_values(self):
        f = open(self.predicting_filename, 'r')
        for segment in self.segments:
            for epoch in segment.epochs:
                epoch.svm_classification = f.readline().strip()
        f.close()

    def evaluate(self):
        TP, FP, TN, FN = (0, 0, 0, 0) 
        for segment in self.segments:
            first_detect = None
            for i, epoch in enumerate(segment.epochs):
                # d=true positive, c=false negtive, a=true negtive, b=false positive
                if segment.classification == '0':
                    if epoch.svm_classification == '0':
                        TN += 1
                    elif epoch.svm_classification == '1':
                        FP += 1
                elif segment.classification == '1':
                    if epoch.svm_classification == '0':
                        FN += 1
                    elif epoch.svm_classification == '1':
                        if first_detect is None: 
                            first_detect = i
                        TP += 1
            # if first_detect is not None:
            #     print first_detect
        print TP, FP, TN, FN
        # TNp = TP + FN     # TNp 所有实际发病
        # TNn = TN + FP     # TNn 所有实际未发病
        sensitive = TP / (TP+FN)
        specificity = TN / (TN+FP)
        accuracy = (TN+TP) / (TP+FP+TN+FN)
        accuracy2 = (sensitive+specificity) / 2
        print 'sensitive = %f\nspecificity = %f\naccuracy=%f\naccuracy2=%f\n' % \
            (sensitive, specificity, accuracy, accuracy2)

    def go(self):
        self.read_true_values()
        self.read_predicted_values()
        self.evaluate()


p = Processor(debug=True)
p.go()
# e = Evaluator('data73.test', 'data73.test.predict')
# e.go()

