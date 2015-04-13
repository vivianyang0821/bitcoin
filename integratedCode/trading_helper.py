

def getRows(dataBase,startInd,endInd):
    import sqlite3
    print(type(dataBase))
    input1 = dataBase+'.rdb'
    conn = sqlite3.connect(input1)
    c = conn.cursor()
    input2 = 'SELECT * FROM '+dataBase+'_ob'
    aa = c.execute(input2)
    rows = []
    count = 0
    for row in aa:
        if (count >= startInd) & (count < endInd):
            rows.append(row)
        if count == endInd:
            break
        count += 1
    return rows
	
def getStartTime(row):
    start = dt.datetime(int(row[5][0:4]), int(row[5][5:7]), int(row[5][8:10]), int(row[5][11:13]), int(row[5][14:16]), int(float(row[5][17:-1])))
    return start
	
def getTimes(rows,startTime):
    import matplotlib.pyplot as plt
    import datetime as dt
    timeSec = [0]
    for afterInd in range(1,len(rows)):
        current = dt.datetime(int(rows[afterInd][5][0:4]), int(rows[afterInd][5][5:7]), int(rows[afterInd][5][8:10]), int(rows[afterInd][5][11:13]), int(rows[afterInd][5][14:16]), int(float(rows[afterInd][5][17:-1])))
        currentDiff = current - startTime
        currentDiffSeconds = currentDiff.days*(24*3600) + currentDiff.seconds
        timeSec.append(currentDiffSeconds)
    return timeSec

def extractTickData(rows,timeSec):
    import numpy as np
    buyPrices = []
    sellPrices = []
    buyTimes = []
    sellTimes = []
    buyQuantity = []
    buyLevels = []
    sellQuantity = []
    sellLevels = []
    for ii in range(len(rows)):
        if rows[ii][1] == 'buy':
            buyPrices.append(rows[ii][2])
            buyTimes.append(timeSec[ii])
            buyQuantity.append(rows[ii][3])
            buyLevels.append(rows[ii][4])
        else:
            sellPrices.append(rows[ii][2])
            sellTimes.append(timeSec[ii])
            sellQuantity.append(rows[ii][3])
            sellLevels.append(rows[ii][4])
    buyPrices = np.array(buyPrices)
    sellPrices = np.array(sellPrices)
    buyTimes = np.array(buyTimes)
    sellTimes = np.array(sellTimes)
    buyQuantity = np.array(buyQuantity)
    buyLevels = np.array(buyLevels)
    sellQuantity = np.array(sellQuantity)
    sellLevels = np.array(sellLevels)
    
    return buyPrices, sellPrices, buyTimes, sellTimes, buyQuantity, buyLevels, sellQuantity, sellLevels	
	
def DHMS(seconds):
    days = np.floor(seconds/3600)
    hours = np.floor((seconds/3600) - days*24)
    minutes = np.floor(((seconds/3600) - days*24 - hours)*60)
    seconds = np.floor((((seconds/3600) - days*24 - hours)*60 - minutes)*60)
    return np.array([days,hours,minutes,seconds])
	
def plotTimes(sellTimes1,sellPrices1,buyTimes1,buyPrices1,sellTimes2,sellPrices2,buyTimes2,buyPrices2,label1,label2,start):
    fig, ax = plt.subplots()
    ax.plot(buyTimes1,buyPrices1,'g-',label=[label1,' Buy Prices'])
    ax.plot(sellTimes1,sellPrices1,'r-',label=[label1,' Sell Prices'])
    ax.plot(buyTimes2,buyPrices2,'b-',label=[label2,' Buy Prices'])
    ax.plot(sellTimes2,sellPrices2,'m-',label=[label2,' Sell Prices'])
    plt.show()
    labels = []
    for ii in range(0,len(ax.get_xticklabels())):
        if ax.get_xticklabels()[ii].get_text() != '':
            timeInc = int(ax.get_xticklabels()[ii].get_text())
            dhms = DHMS(timeInc)
            step = dt.timedelta(days = dhms[0],hours = dhms[1],minutes = dhms[2],seconds = dhms[3]) 
            currentLabel = start + step
            labels.append(currentLabel.strftime('%Y:%m:%d:%H:%M:%S'))
        else:
            labels.append(ax.get_xticklabels()[ii].get_text())
    fig, ax = plt.subplots()
    ax.plot(buyTimes1,buyPrices1,'g-',label=[label1,' Buy Prices'])
    ax.plot(sellTimes1,sellPrices1,'r-',label=[label1,' Sell Prices'])
    ax.plot(buyTimes2,buyPrices2,'b-',label=[label2,' Buy Prices'])
    ax.plot(sellTimes2,sellPrices2,'m-',label=[label2,' Sell Prices'])
    ax.set_xticklabels(labels)
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Buy and Sell Prices')
    plt.title([label1,' and ',label2,' Bitcoin Bid/Ask'])
    plt.legend()
    plt.show()

def timeFilterData(data,times,cutoff1,cutoff2):
    filteredData = data[(cutoff1 < times) & (times < cutoff2)]
    return filteredData

def trainData(sellTimes1, sellPrices1, buyTimes1, buyPrices1, sellTimes2, sellPrices2, buyTimes2, buyPrices2):
    import numpy as np
    from scipy import interpolate
    from sklearn import linear_model
    
    tck = interpolate.splrep(sellTimes1, sellPrices1, s=0)
    compare_buy_1 = interpolate.splev(buyTimes1, tck, der=0)
    compare_buy_2 = interpolate.splev(buyTimes2, tck, der=0)
    compare_sell_2 = interpolate.splev(sellTimes2, tck, der=0)
    
    clfT = linear_model.LinearRegression()
    formatX = np.transpose(sellPrices1)
    formatY = np.transpose(buyPrices2)
    newX = np.array([formatX]).T
    clfT.fit(newX, formatY)
    slopeToUse_1 = clfT.coef_
    interceptToUse_1 = clfT.intercept_

    clfT2 = linear_model.LinearRegression()
    formatX = np.transpose(buyPrices1)
    formatY = np.transpose(sellPrices2)
    newX = np.array([formatX]).T
    clfT2.fit(newX, formatY)
    slopeToUse_2 = clfT2.coef_
    interceptToUse_2 = clfT2.intercept_
    
    return slopeToUse_1, interceptToUse_1, slopeToUse_2, interceptToUse_2

def trade(timeLine,sellPrices1,buyPrices1,sellPrices2,buyPrices2,zScore,slope1,int1,slope2,int2,time_ref,sellPrices1_ref,buyPrices1_ref,sellPrices2_ref,buyPrices2_ref):
 
    bank1 = 10**4
    bank2 = 10**4
    coins1 = 10
    coins2 = 10

    bankTracker1 = [bank1]
    coinTracker1 = [coins1]
    naiveWorthTracker1 = [bank1 + coins1*buyPrices1[0]]

    bankTracker2 = [bank2]
    coinTracker2 = [coins2]
    naiveWorthTracker2 = [bank2 + coins2*buyPrices2[0]]

    zScoreTracker_1 = []
    zScoreTracker_2 = []

    extremeFlag = 0
    betSize = 1

    for ii in range(0,len(timeLine)):
        currentTime = timeLine[ii]
        sellP1 = sellPrices1[ii]
        sellP2 = sellPrices2[ii]
        buyP1 = buyPrices1[ii]
        buyP2 = buyPrices2[ii]

        sellPrices1_past = sellPrices1_ref[time_ref < currentTime]
        buyPrices2_past = buyPrices2_ref[time_ref < currentTime]
        time_past = time_ref[time_ref < currentTime]
        time_filt = time_past - time_past[-1]
        
        sellPrices1_past = sellPrices1_past[time_filt > -3600*12]
        buyPrices2_past = buyPrices2_past[time_filt > -3600*12]

        standardDev_1 = np.std((slope1*sellPrices1_past + int1) - buyPrices2_past)
        meanHist_1 = np.mean((slope1*sellPrices1_past + int1) - buyPrices2_past)

        zScore_1 = (((slope1*sellP1 + int1) - buyP2) - meanHist_1)/standardDev_1
        zScoreTracker_1.append(zScore_1)

        buyPrices1_past = buyPrices1_ref[time_ref < currentTime]
        sellPrices2_past = sellPrices2_ref[time_ref < currentTime]

        buyPrices1_past = buyPrices1_past[time_filt > -3600*12]
        sellPrices2_past = sellPrices2_past[time_filt > -3600*12]

        standardDev_2 = np.std((slope2*buyPrices1_past + int2) - sellPrices2_past)
        meanHist_2 = np.mean((slope2*buyPrices1_past + int2) - sellPrices2_past)

        zScore_2 = (((slope2*buyP1 + int2) - sellP2) - meanHist_2)/standardDev_2
        zScoreTracker_2.append(zScore_2)

        if zScore_1 < -2:
            bank2 = bank2 + betSize*buyP2
            coins2 = coins2 - betSize
            bank1 = bank1 - betSize*sellP1
            coins1 = coins1 + betSize
            extemeFlag = -1
        elif zScore_2 > 2:
            bank1 = bank1 + betSize*buyP1
            coins1 = coins1 - betSize
            bank2 = bank2 - betSize*sellP2
            coins2 = coins2 + betSize
            extremeFlag = 1
        elif (abs(zScore_1) < .5) & (abs(zScore_2) < .5):
         
            if extremeFlag == -1 | extremeFlag == 1:
                if coins1 > 0:
                    bank1 = bank1 + coins1*buyP1
                    bank1 = bankSTAMP - 0.002*coins1*buyP1
                else:
                    bank1 = bank1 + coins1*sellP1
                    bank1 = bank1 + 0.002*coins1*sellP1
                if coins2 > 0:
                    bank2 = bank2 + coins2*buyP2
                    bank2 = bank2 - 0.002*coins2*buyP2
                else:
                    bank2 = bank2 + coins2*sellP2 
                    bank2 = bank2 + 0.002*coins2*sellP2
                coins1 = 0
                coins2 = 0
                extremeFlag = 0

        bankTracker1.append(bank1)
        coinTracker1.append(coins1)
        naiveWorthTracker1.append(10**4 + 10*buyP1)

        bankTracker2.append(bank2)
        coinTracker2.append(coins2)
        naiveWorthTracker2.append(10**4 + 10*buyP2)

    bankTracker1 = np.array(bankTracker1)
    coinTracker1 = np.array(coinTracker1)
    naiveWorthTracker1 = np.array(naiveWorthTracker1)

    bankTracker2 = np.array(bankTracker2)
    coinTracker2 = np.array(coinTracker2)
    naiveWorthTracker2 = np.array(naiveWorthTracker2)
    
    return bankTracker1, naiveWorthTracker1, bankTracker2, naiveWorthTracker2
	
def plotReturn(timeLine,bankTracker1,naiveWorthTracker1,bankTracker2,naiveWorthTracker2,startTime,label1,label2):
    fig, ax = plt.subplots()
    ax.plot(timeLine,bankTracker1[1:],'g-',label=[label1,' Portfolio Pairs Trading'])
    ax.plot(timeLine,10**4*np.ones(len(naiveWorthTracker1[1:])),'r-',label=u'Portfolio Inactive')
    ax.plot(timeLine,bankTracker2[1:],'b-',label=[label2,' Portfolio Pairs Trading'])
    ax.plot(timeLine,bankTracker1[1:]+bankTracker2[1:],'c-',label=u'Total Portfolio Pairs Trading')
    ax.plot(timeLine,2*10**4*np.ones(len(naiveWorthTracker1[1:])),'y-',label=u'Total Inactive Trading')
    plt.show()
    labels = []
    for ii in range(0,len(ax.get_xticklabels())):
        if ax.get_xticklabels()[ii].get_text() != '':
            timeInc = int(ax.get_xticklabels()[ii].get_text())
            dhms = DHMS(timeInc)
            step = dt.timedelta(days = dhms[0],hours = dhms[1],minutes = dhms[2],seconds = dhms[3]) 
            currentLabel = startTime + step
            labels.append(currentLabel.strftime('%Y:%m:%d:%H:%M:%S'))
        else:
            labels.append(ax.get_xticklabels()[ii].get_text())
    fig, ax = plt.subplots()
    ax.plot(timeLine,bankTracker1[1:],'g-',label=[label1,' Portfolio Pairs Trading'])
    ax.plot(timeLine,10**4*np.ones(len(naiveWorthTracker1[1:])),'r-',label=u'Portfolio Inactive')
    ax.plot(timeLine,bankTracker2[1:],'b-',label=[label2,' Portfolio Pairs Trading'])
    ax.plot(timeLine,bankTracker1[1:]+bankTracker2[1:],'c-',label=u'Total Portfolio Pairs Trading')
    ax.plot(timeLine,2*10**4*np.ones(len(naiveWorthTracker1[1:])),'y-',label=u'Total Inactive Trading')
    ax.set_xticklabels(labels)
    plt.xlabel('Time')
    plt.ylabel('Account Value')
    plt.title([label,' and ',label2,' Pairs Trading'])
    plt.legend(loc = 0)
    plt.show()	
	
	