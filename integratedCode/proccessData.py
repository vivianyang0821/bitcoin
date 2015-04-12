#-------------------------------------------------------------------------------
# Name:        proccessData
# Purpose:
#
# Author:      Wenshuai Ye
#
# Created:     11/04/2015
# Copyright:   (c) Superman 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import sqlite3
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

class processData:
    def __init__(self, filename, tablename, split = True, extractTime = True):
        self.data = self._getData(filename, tablename, split, extractTime)

    ### Should be protected ###
    def _getData(self, filename, tablename, split, extractTime):
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        Dict = []
        sql_command = 'SELECT * FROM ' + tablename
        for row in c.execute(sql_command):
            Dict.append(row)
        data = pd.DataFrame(Dict)
        data.columns = ['index','buy/sell','price','size','ob_level','timestamp']
        data['timestamp'] = pd.to_datetime(data.timestamp)
        # Timestamp extraction
        if extractTime:
            data["year"] = map(lambda x: x.year, data.timestamp)
            data["month"] = map(lambda x: x.month, data.timestamp)
            data["day"] = map(lambda x: x.day, data.timestamp)
            data["hour"] = map(lambda x: x.hour, data.timestamp)
            data["minute"] = map(lambda x: x.minute, data.timestamp)
            data["second"] = map(lambda x: x.second, data.timestamp)
        if split:
            bid = data[(data['buy/sell'] == 'buy')]
            ask = data[(data['buy/sell'] == 'sell')]
            return bid,ask
        else:
            return data

    # Visualize the price trend and the spread between bid and ask overtime.
    def graphRawPrice(self):
        assert type(self.data == tuple)
        bid, ask = self.data[0], self.data[1]
        fig = plt.figure(figsize=(10,7))
        ax1 = plt.subplot2grid((40,40),(0,0),rowspan=40,colspan=40)

        ax1.plot(range(len(bid)),bid['price'], label="bid")
        ax1.plot(range(len(ask)),ask['price'], label="ask")
        ax1.legend()
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        ax1_2 = ax1.twinx()

        ax1_2.fill_between(range(bid.shape[0]),0,(np.array(ask['price'])
                            -np.array(bid['price'])),
                            facecolor='g', alpha=.3)

        plt.subplots_adjust(bottom=.23)
        plt.grid(True)
        plt.xlabel("Timeline")
        plt.show()

    # Generate auto-correlation plot.
    def graphAutoCorr(self, variable, Maxlags = 20, Normed = True, Usevlines=True):
        plt.acorr(variable - np.mean(variable), maxlags=Maxlags,  normed=Normed, usevlines=Usevlines);
        plt.xlabel("Lag")
        plt.ylabel("Auto Correlation")
        plt.title("Auto Correlation Plot")
        plt.xlim([0,Maxlags])
        plt.show()

    def checkNormality(self, variable):
        pass

    def checkStationarity(self, variable):
        pass
