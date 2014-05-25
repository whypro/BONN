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

    def __repr__(self):
        string = '\nclassification: ' + self.classification
        string += '\nrecords: \n' + '\n'.join([str(point) for point in self.records])
        string += '\nepochs: \n' + '\n'.join([str(epoch) for epoch in self.epochs])
        return string

class Epoch(object):
    def __init__(self):
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
        for segment in self.segments:
            epoch_num = int((len(segment.records)-point_num*overlap) / (point_num*(1-overlap)))     # 默认条件下为 31
            epochs = []
            for epoch_i in range(1, epoch_num+1):
                begin = int(point_num*overlap*(epoch_i-1))
                end = int(point_num*overlap*(epoch_i-1)+point_num)
                # print epoch_i, ',', begin, ':', end
                epoch = segment.records[begin:end]
                epochs.append(epoch)
            segment.epochs = epochs
        # print len(self.overlapped_segments)
        if self.debug:
            print 'segments length = %d, segments[0].epochs length = %d, segments[0].epochs[0] length = %d.' % (
                len(self.segments), 
                len(self.segments[0].epochs), 
                len(self.segments[0].epochs[0])
            )

    def go(self):
        missions = {
            'A/Z.zip': 'A/Z.txt', 
            #'D/F.zip': 'D/F.txt', 
            #'E/S.zip': 'E/S.txt',
        }
        for i, o in missions.items():
            self.read_from_zipfile(i)
            self.calc_epoch()

        # self.reset()
        # self.merge(missions.values(), 'data')


p = Processor(debug=True)
p.go()
f = open('records.txt', 'w')
f.write(str(p.segments))
f.close()
