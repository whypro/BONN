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

    def __convert_to_libsvm_data(self, segment, base):
        """
        base in ['dec', 'bin']
        <Classification> 1:<cA3 Average> 2:<cA3 Variance> 3:<cD3 Average> 4:<cD3 Variance> 5:<cD2 Average> 6:<cD2 Variance> 7:<cD1 Average> 8:<cD1 Variance>
        """
        data = ''
        for epoch in segment.epochs:
            row = [segment.classification]
            if base == 'dec':
                for i, feature in enumerate(epoch.features, start=1):
                    row.append('%d:%f' % (i, feature))
            elif base == 'bin':
                for i, feature in enumerate(epoch.features, start=1):
                    row.append('%d:%s' % (i, self.dec2bin(feature)))
            else:
                raise
            data += ' '.join(row)
            data += '\n'
        return data

    def save_train_and_test_data(self, training_filename='data', predicting_filename='data.test', base='dec'):
        training_segments = self.segments[:2*len(self.segments)//3]
        predicting_segments = self.segments[2*len(self.segments)//3:]

        training_file = open(training_filename, 'w')
        for segment in training_segments:
            data = self.__convert_to_libsvm_data(segment, base)
            training_file.write(data)
        training_file.close()
        print 'training data saved to %s by %s' % (training_filename, base)

        predicting_file = open(predicting_filename, 'w')
        for segment in predicting_segments:
            data = self.__convert_to_libsvm_data(segment, base)
            predicting_file.write(data)
        predicting_file.close()
        print 'predicting data saved to %s by %s' % (predicting_filename, base)

    def save_to_file(self, filename="output.txt"):
        """
        1:<cA3 Average> 2:<cA3 Variance> 3:<cD3 Average> 4:<cD3 Variance> 5:<cD2 Average> 6:<cD2 Variance> 7:<cD1 Average> 8:<cD1 Variance>
        """
        f = open(filename, 'w')
        for segment in self.segments:
            for epoch in segment.epochs:
                row = [segment.classification]
                for i, feature in enumerate(epoch.features, start=1):
                    row.append('%d:%f' % (i, feature))
                f.write(' '.join(row))
                f.write('\n')
            f.write('\n')
        f.close()
        if self.debug:
            print 'data saved to %s.' % filename

    @staticmethod
    def dec2bin(number):
        negative = number < 0
        integer_part = abs(int(number))
        fraction_part = abs(number) - integer_part

        print negative
        print integer_part
        print fraction_part

        # 整数部分转为原码
        ip_bin = ''
        for i in range(15, -1, -1):
            ip_bin += str(integer_part >> i & 1)
        print ip_bin

        # 小数部分转为原码
        fp_bin = ''
        temp = fraction_part
        for i in range(16):
            temp *= 2
            fp_bin += str(int(temp))
            temp -= int(temp)
        print fp_bin
        result = ip_bin + fp_bin

        # 原码转为补码
        return bin(Processor.str2bin(result, negative))[2:].zfill(32)


    @staticmethod
    def str2bin(string, negative):
        binary = 0
        print string
        if negative:
            binary = 1
        binary = binary << 1
        for c in string[1:]:
            if c != '0': binary = binary | 1
            binary = binary << 1
        print binary
        if negative: binary = -binary
        binary = binary & 0xffffffff
        return binary

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
        self.save_train_and_test_data('data', 'data.test', base='dec')
        self.save_train_and_test_data('data.bin', 'data.bin.test', base='bin')

        # self.reset()
        # self.merge(missions.values(), 'data')


class Evaluator(object):
    def __init__(self):
        self.segments = []

    def read_true_values(self, filename='data.test', segment_num = 100, epoch_num=31):
        f = open(filename, 'r')
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

    def read_predicted_values(self, filename='data.out'):
        f = open(filename, 'r')
        for segment in self.segments:
            for epoch in segment.epochs:
                epoch.svm_classification = f.readline().strip()
        f.close()

    def evaluate(self):
        a, b, c, d = (0, 0, 0, 0)
        for segment in self.segments:
            first_detect = None
            for i, epoch in enumerate(segment.epochs):
                # d=true positive, c=false negtive, a=true negtive, b=false positive
                if segment.classification == '0':
                    if epoch.svm_classification == '0':
                        a += 1
                    elif epoch.svm_classification == '1':
                        c += 1
                elif segment.classification == '1':
                    if epoch.svm_classification == '0':
                        b += 1
                    elif epoch.svm_classification == '1':
                        if first_detect is None:
                            first_detect = i
                        d += 1
            # if first_detect is not None:
            #     print first_detect
        print a, b, c, d
        TP = d          # True Positive
        TN = a          # True Negative
        TNp = c + d     # TNp 所有实际发病
        TNn = a + b     # TNn 所有实际未发病
        sensitive = TP / TNp
        specificity = TN / TNn
        accuracy = (a+d) / (a+b+c+d)
        accuracy2 = (sensitive+specificity) / 2
        print 'sensitive = %f\nspecificity = %f\naccuracy=%f\naccuracy2=%f\n' % \
            (sensitive, specificity, accuracy, accuracy2)

    def go(self):
        self.read_true_values('data7.test')
        self.read_predicted_values('data7.out')
        self.evaluate()


p = Processor(debug=True)
print p.dec2bin(-1.5)
# p.go()
# e = Evaluator()
# e.go()
# f = open('records.txt', 'w')
# f.write(str(p.segments))
# f.close()
