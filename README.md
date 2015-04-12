![alt tag](Wenshuai/img/bitcoin.png)
## ac297r Capstone Project: Bitcoin Research
Team Member: Daniel Rajchwald, Wenwan Yang, Wenshuai Ye
####Introduction
Bitcoin is developed by an anonymous hacker under the name Satoshi Nakamoto in 2009. It is a completely decentralized peer-to-peer system governed by open sourced algorithms. Bitcoin accounts are anonymous. They are just public key addresses without any personally identifiable information attached. Unlike other currencies, bitcoin has a maximum circulation of just under 21 million coins which cause the value of each coin to increase over time. Every transaction is verified by the entire Bitcoin network which makes it nearly impossible to counterfeit a Bitcoin. The motivation of this project is to implement some trading strategies and find the arbitrage opportunities of bitcoin. Some trading algorithms will be introduced in the following part. There are two ways to find the arbitrage opportunities. The first one is to buy a dual listed stock at a lower price in one market and simultaneously selling it at a higher price in another market. The second is the cross currency arbitrage which is an act of exploiting an arbitrage opportunity resulting from a pricing discrepancy among three different currencies in the foreign exchange currencies. We got the tick level data from Morgan Stanley and it is used to test performance of each algorithm.

####Trading Algorithms
We tried various standard trading algorithms before building more sophisticated statistical models. These algorithms include momentum trading, pairs trading, backtesting, etc., which are simple and popular strategies that have been around for decades. By implementing all the strategies on one single dataset, we can compare the results and conclude which algorithms should be used in a given market. With these exploratory analyses done, our future work will focus on incorporating the bitcoin properties we have learned so far to build more sophisticated statistical models such as a Hidden Markov Model (HMM). HMM is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states. Our current thought is to model the intrinsic values of bitcoin as the unobserved state and the actual prices as the observed state. The trade signal relies on the current distribution of the intrinsic values. Depending on the performance, we might adjust our model when we implement it.

####Bitcoin Properties Learned through Trading and Ensemble Approach
After comparing algorithm performance across different market conditions, we hope to learn unique properties of Bitcoin such as whether the log-normal distribution is a good approximation to Bitcoin��s probability distribution and whether Bitcoin��s high volatility allows for basic strategies like momentum trading to be more profitable. Another goal is to classify BTC price timeseries as suitable for specific algorithms. For instance, if the previous 50 BTC prices show a slope of 10BTC/hour and the current price is 2 standard deviations above the moving average, than momentum trading will have an empirical return greater than that of pairs trading. This requires substantial feature engineering but fortunately there are numerous metrics accrued through the trading strategies we applied that can be used to classify market conditions such as moving averages, stochastic oscillators, and cointegration. We hope to implement an ensemble trading approach that applies the best trading strategy or combination of trading strategies for a given history (price_BTCt-n, ..., price_BTCt-1) and/or market condition. The ensemble strategy will be tested for robustness to verify if the BTC price timeseries classifier is effective.

classify market conditions such as moving averages, stochastic oscillators, and cointegration. We hope to implement an ensemble trading approach that applies the best trading strategy or combination of trading strategies for a given history (price_BTCt-n, ..., price_BTCt-1) and/or market condition. The ensemble strategy will be tested for robustness to verify if the BTC price timeseries classifier is effective.

