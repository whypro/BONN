# -*- coding: utf-8 -*-
from __future__ import division
import os
import pywt
import numpy as np
import zipfile

class DWT(object):
    CLASS_MAP = {'A': '1', 'D': '1', 'E': '-1'}

    def __init__(self, debug=False):
        self.debug = debug

        self.dirname = ''

        self.record = []
        self.epoch = []
        self.dwt_epoch = []
        self.features = []
    
    def reset(self):
        self.record = []
        self.epoch = []
        self.dwt_epoch = []
        self.features = []
    
    def read_from_zipfile(self, filename):
        z = zipfile.ZipFile(os.path.join(self.dirname, filename), 'r', )
        for fname in z.namelist():
            f = z.open(fname, 'r')
            for line in f:
                point = line.strip()
                if point:
                    self.record.append(int(point))
            f.close()
        if self.debug:
            print '%d records has been read.' % len(self.record)
            # print self.record

    def read_from_file(self, filename):
        f = open(os.path.join(self.dirname, filename), 'r')
        for line in f:
            point = line.strip()
            if point:
                self.record.append(int(point))
        f.close()
        if self.debug:
            print '%d records has been read.' % len(self.record)
            print self.record


    def calc_epoch(self, point=256, overlap=1/2):
        epoch_num = int((len(self.record)-point*overlap) / (point*(1-overlap)))
        for epoch_i in range(1, epoch_num+1):
            #print('[%d]')
            begin = int(point*overlap*(epoch_i-1))
            end = int(point*overlap*(epoch_i-1)+point)
            # print epoch_i, ',', begin, ':', end
            self.epoch.append(self.record[
                begin : end
            ])
        if self.debug:
            print 'epoch = %d, point = %d.' % (len(self.epoch), len(self.epoch[0]))
            # print self.epoch[0][128], self.epoch[1][0] 

    def discrete_wavelet_transform(self):
        for epoch in self.epoch:
            result = pywt.wavedec(epoch, 'db4', level=4)
            self.dwt_epoch.append(result)
        if self.debug:
            print 'dwt result length = %d.' % len(self.dwt_epoch[0])

        
    def extract_features(self):
        for dwt_epoch in self.dwt_epoch:
            feature = []
            for i in dwt_epoch:
                average = self.calc_average(i)
                variance = self.calc_variance(i)
                #feature.append(average)
                feature.append(variance)
            self.features.append(feature)
        if self.debug:
            print '%d features in every epoch.' % len(self.features[0])

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

    def save_to_file(self, filename):
        """
        1:<cA3 Average> 2:<cA3 Variance> 3:<cD3 Average> 4:<cD3 Variance> 5:<cD2 Average> 6:<cD2 Variance> 7:<cD1 Average> 8:<cD1 Variance>
        """
        f = open(os.path.join(self.dirname, filename), 'w')
        for feature in self.features:
            for key, value in self.CLASS_MAP.items():
                if filename[0] == key:
                    classification = value
            row = [classification]
            for i in range(0, len(feature)):
                row.append('%d:%f' % (i+1, feature[i]))
            f.write(' '.join(row))
            f.write('\n')
        f.close()
        if self.debug:
            print 'data saved to %s.' % filename

    def merge(self, file_list, merged_filename):
        print file_list
        merged_file = open(os.path.join(self.dirname, merged_filename), 'w')
        for f_name in file_list:
            f = open(f_name, 'r')
            merged_file.write(f.read())
            f.close()
        merged_file.close()




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
            self.save_to_file(o)
            self.reset()
        self.merge(missions.values(), 'data')

dwt = DWT(debug=True)
dwt.go()




