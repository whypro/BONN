# -*- coding: utf-8 -*-
from __future__ import division
import os
import pywt
import numpy as np
import zipfile
import random


class DWT(object):
    """读取 A.zip, B.zip, C.zip 后进行小波变换，特征提取后保存至 A.txt, B.txt, C.txt。"""
    CLASS_MAP = {'A': '0', 'D': '0', 'E': '1'}

    def __init__(self, debug=False):
        self.debug = debug

        self.dirname = ''

        self.segments = []    # 记录集，以 segment 为单位，一个文件即为一个 segment
        self.overlapped_segments = []   # 交叠后的 segments
        # self.epoch = []     # epoch 集，以 segment 为单位
        self.dwt_segments = [] # 离散小波变换后的 segments
        self.feature_segments = []  # 特征提取后的向量集
    
    def reset(self):
        self.segments = []
        self.overlapped_segments = []
        self.dwt_segments = []
        self.feature_segments = []
    
    def read_from_zipfile(self, filename):
        z = zipfile.ZipFile(os.path.join(self.dirname, filename), 'r', )
        sum_points = 0  # segments 中 point 的个数
        for fname in z.namelist():
            f = z.open(fname, 'r')
            segment = []
            for line in f:
                point = line.strip()
                if point:
                    segment.append(int(point))
                    sum_points += 1
            self.segments.append(segment)
            f.close()
        if self.debug:
            print '%d points has been read.' % sum_points
            print '%d segment has been read.' % len(self.segments)
            # print self.segments

    def read_from_file(self, filename):
        f = open(os.path.join(self.dirname, filename), 'r')
        for line in f:
            point = line.strip()
            if point:
                self.segments.append(int(point))
        f.close()
        if self.debug:
            print '%d records has been read.' % len(self.segments)
            print self.segments

    def calc_epoch(self, point_num=256, overlap=1/2):
        # segments -> segment(epochs) -> epoch(points)
        for segment in self.segments:
            epoch_num = int((len(segment)-point_num*overlap) / (point_num*(1-overlap)))     # 默认条件下为 31
            epochs = []
            for epoch_i in range(1, epoch_num+1):
                begin = int(point_num*overlap*(epoch_i-1))
                end = int(point_num*overlap*(epoch_i-1)+point_num)
                # print epoch_i, ',', begin, ':', end
                epoch = segment[begin:end]
                epochs.append(epoch)
            self.overlapped_segments.append(epochs)
        # print len(self.overlapped_segments)
        if self.debug:
            print 'segments length = %d, epochs length = %d, points length = %d.' % (
                len(self.overlapped_segments), 
                len(self.overlapped_segments[0]), 
                len(self.overlapped_segments[0][0])
            )
            # f = open('overlapped_segments.txt', 'w')
            # for segment in self.overlapped_segments:
            #     for epochs in segment:
            #         for epoch in epochs:
            #             f.write(str(epoch)+' ')
            #         f.write('\n')
            #     f.write('\n')

    def discrete_wavelet_transform(self):
        # segments -> segment(epochs) -> results
        for epochs in self.overlapped_segments:
            dwt_epochs = []
            for epoch in epochs:
                dwt_results = pywt.wavedec(epoch, 'db4', level=3)
                dwt_epochs.append(dwt_results)
            self.dwt_segments.append(dwt_epochs)
        if self.debug:
            print 'dwt segments length = %d, dwt epochs length = %d, dwt results length = %d.' % (
                len(self.dwt_segments), 
                len(self.dwt_segments[0]), 
                len(self.dwt_segments[0][0]),
            )

        
    def extract_features(self):
        # segments -> epochs -> features
        for dwt_epochs in self.dwt_segments:
            feature_epochs = []
            for dwt_results in dwt_epochs:
                features = []
                for i in dwt_results:
                    average = self.calc_average(i)
                    variance = self.calc_variance(i)
                    features.append(average)
                    features.append(variance)
                feature_epochs.append(features)
            self.feature_segments.append(feature_epochs)

        if self.debug:
            print 'feature segments length = %d, feature epochs length = %d, feature features length = %d.' % (
                len(self.feature_segments), 
                len(self.feature_segments[0]), 
                len(self.feature_segments[0][0]),
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
    def calc_abs(data, average=None):
        if not average:
            average = sum(data)/len(data)
        diff = []
        for point in data:
            diff.append(abs(point-average))
        return sum(diff)/len(diff)

    def shuffle_segments(self):
        random.shuffle(self.feature_segments)

    def save_to_file(self, filename):
        """
        1:<cA3 Average> 2:<cA3 Variance> 3:<cD3 Average> 4:<cD3 Variance> 5:<cD2 Average> 6:<cD2 Variance> 7:<cD1 Average> 8:<cD1 Variance>
        """
        f = open(os.path.join(self.dirname, filename), 'w')
        # fr = open(os.path.join(self.dirname, filename+'.range'), 'w')
        for feature_epochs in self.feature_segments:
            for features in feature_epochs:
                classification = self.CLASS_MAP[filename[0]]
                row = [classification]
                for i, feature in enumerate(features):
                    row.append('%d:%f' % (i+1, feature))
                f.write(' '.join(row))
                f.write('\n')
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
        # self.merge(missions.values(), 'data')

dwt = DWT(debug=True)
dwt.go()




