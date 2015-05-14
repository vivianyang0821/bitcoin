#-------------------------------------------------------------------------------
# Name:        pattern Recognition Technique
# Purpose:
#
# Author:      Wenshuai Ye
#
# Created:     12/04/2015
# Copyright:   (c) Superman 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from proccessData import processData
import numpy as np
import pandas as pd
import sqlite3
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

class PatternRecognition:
    def __init__(self, filename, tablename):
        process = processData()
        self.data = process.getData(filename, tablename)
        self.patternAr = []
        self.performanceAr = []
        self.avgLine = (np.array(data[0]['price'])+np.array(data[1]['price']))/2

    def percentChange(self, startPoint, currentPoint):
        return ((float(currentPoint)-startPoint)/abs(startPoint))*100.00

    def patternStorage(self, patternLength):
        patStartTime = time.time()

        x = len(self.avgLine) - patternLength

        y = patternLength + 1

        while y < x:
            pattern = []
            for i in xrange(1,patternLength + 1):

                pattern.append(self.percentChange(self.avgLine[y - patternLength - 1],
                                              self.avgLine[y - (patternLength + 1) + i]))
                '''
                pattern.append(self.percentChange(self.avgLine[y - (patternLength + 1) + i],
                                              self.avgLine[y]))
                '''
            outcomeRange = self.avgLine[y+20:y+30]
            currentPoint = self.avgLine[y]

            try:
                avgOutcome = reduce(lambda x, y: x+y,
                                    outcomeRange / len(outcomeRange))
            except Exception, e:
                print str(e)
                avgOutcome=0


            futureOutcome = self.percentChange(currentPoint, avgOutcome)
            self.patternAr.append(pattern)
            self.performanceAr.append(futureOutcome)
            y += 1

        patEndTime = time.time()
        print len(self.patternAr)
        print len(self.performanceAr)
        print "Pattern storage took:", patEndTime - patStartTime, " seconds"

    def currentRecognition(self, startIndex, patternLength):
        patForRec = []
        for i in xrange(1,patternLength+1):

            patForRec.append(percentChange(self.avgLine[-startIndex],
                                           self.avgLine[-startIndex+i]))


        return patForRec

    def simPatternRecognition(self, threshold, patForRec):
        plt.figure()
        #Use similarity to find patterns
        i = 0

        plt.plot(np.arange(0,len(patForRec)), patForRec)

        predictions = []
        for eachPattern in self.patternAr:
            howSim = 0
            for i in xrange(len(patForRec)):
                howSim += 100.0 - abs(self.percentChange(eachPattern[i],
                                                 patForRec[i]))
            howSim = howSim / len(patForRec)
            if howSim > threshold and howSim < 100:
                patdex = self.patternAr.index(eachPattern)
                '''
                print "##########################"
                print "##########################"
                '''
                plt.plot(np.arange(0,len(patForRec)), eachPattern)
                plt.scatter([55],self.performanceAr[patdex],s=60)

                predictions.append(self.performanceAr[patdex])
                '''
                print patForRec
                print "=========================="
                print "=========================="
                print eachPattern
                print "--------------------------"
                print "predicted outcome", performanceAr[patdex]
                print "##########################"
                print "##########################"
                '''

        plt.grid(True)
        plt.xlim([0,60])
        plt.show()
        plt.hist(predictions,bins=25)
        plt.show()

        if len(predictions) == 0:
            return 0
        else:
            return np.mean(predictions)
