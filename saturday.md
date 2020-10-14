News Popularity Saturday Data
================
Shuang Du
10/16/2020

## Load Libraries

``` r
library(readxl);
library(tidyverse);
library(caret);
library(modelr);
library(rpart);
library(kableExtra);
```

## Read in Data

``` r
getData <- function(day) {

  newsPopData <- read_csv("../../raw_data/OnlineNewsPopularity.csv")
  
  if (day == 'monday') {
    newsPopData <- newsPopData %>% filter(weekday_is_monday == 1)
  } else if(day == 'tuesday') {
    newsPopData <- newsPopData %>% filter(weekday_is_tuesday == 1)
  } else if(day == 'wednesday') {
    newsPopData <- newsPopData %>% filter(weekday_is_wednesday == 1)
  } else if(day == 'thursday') {
    newsPopData <- newsPopData %>% filter(weekday_is_thursday == 1)
  } else if(day == 'friday') {
    newsPopData <- newsPopData %>% filter(weekday_is_friday == 1)
  } else if(day == 'saturday') {
    newsPopData <- newsPopData %>% filter(weekday_is_saturday == 1)
  } else if(day == 'sunday') {
    newsPopData <- newsPopData %>% filter(weekday_is_sunday == 1)
  } else {
    stop("Invalid date")
  }
  return(newsPopData)
}

newsPopData <- getData(params$day)
```

## Set Aside Training Data

``` r
set.seed(92)
trainIndex <- createDataPartition(newsPopData$shares, 
                                  p = 0.7, list = FALSE)

newsPopTrain <- newsPopData[as.vector(trainIndex),];
newsPopTest <- newsPopData[-as.vector(trainIndex),];
```

## Center and Scale

``` r
preProcValues <- preProcess(newsPopTrain, method = c("center", "scale"))
newsPopTrain <- predict(preProcValues, newsPopTrain) 
newsPopTest <- predict(preProcValues, newsPopTest)
```

## Summary of a Few Variables

The plots below show a histogram of the number of shares for the given
day. Scatter plots on the effect of max positive polarity, article time
delta and number of videos in the article are also included.

As expected the histogram has a strong right tail, as seem by the
summary stats which show a very high maximum and a median severals
orders of magnitude lower. This is expected for because of the “viral”
nature of online popularity.

``` r
summary(newsPopTrain$shares)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -0.25079 -0.17423 -0.13160  0.00000 -0.03416 37.37878

``` r
g0 <- ggplot(newsPopTrain, aes(x=shares))
g0 + geom_histogram(binwidth = 0.5) + ggtitle('Histogram for Number of Shares') + ylab('Number of Shares') + xlab('Shares')
```

![](saturday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
summary(newsPopTrain$max_positive_polarity)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -3.09122 -0.71581  0.07599  0.00000  0.86780  0.86780

``` r
g1 <- ggplot(newsPopTrain, aes(x = max_positive_polarity, y = shares )) 
g1 + geom_point() + ggtitle('Scatter of Max Positive Polarity Effect') + ylab('Shares') + xlab('Max Positive Polarity')
```

![](saturday_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

``` r
summary(newsPopTrain$timedelta)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -1.60493 -0.94480 -0.02062  0.00000  0.87055  1.76173

``` r
g2 <- ggplot(newsPopTrain, aes(x = timedelta, y = shares )) 
g2 + geom_point() + ggtitle('Scatter of Article Age Effect') + ylab('Shares') + xlab('Time Delta')
```

![](saturday_files/figure-gfm/unnamed-chunk-5-3.png)<!-- -->

``` r
summary(newsPopTrain$num_videos)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -0.28543 -0.28543 -0.28543  0.00000 -0.04527 17.48692

``` r
g3 <- ggplot(newsPopTrain, aes(x = num_videos, y = shares )) 
g3 + geom_point() + ggtitle('Scatter of Videos Number Effect') + ylab('Shares') + xlab('Number of Videos')
```

![](saturday_files/figure-gfm/unnamed-chunk-5-4.png)<!-- -->

## Modeling

### Standard Tree Based Model (no ensemble)

The type of model being fitted here is a decision tree. The tree splits
are based on minimizing the residual sum of squares for each region.

``` r
rpartFit <- train(shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens
                 + num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + num_keywords + data_channel_is_lifestyle +
                 data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + data_channel_is_world +
                 self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity
                 + global_rate_positive_words + global_rate_negative_words + rate_positive_words + rate_negative_words + avg_positive_polarity +
                  min_positive_polarity + max_positive_polarity + avg_negative_polarity + min_negative_polarity + max_negative_polarity + title_subjectivity
                 + title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity, data = newsPopTrain,
             method = "rpart",
             trControl = trainControl(method = "cv", number = 10))
rpartFit
```

    ## CART 
    ## 
    ## 1719 samples
    ##   37 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1547, 1546, 1546, 1549, 1547, 1546, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp           RMSE       Rsquared     MAE      
    ##   0.005336585  0.8098810  0.009495701  0.2487085
    ##   0.005628144  0.8092891  0.009625068  0.2483993
    ##   0.055842740  0.7961987  0.013008413  0.2400282
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.05584274.

``` r
# create the prediction
pred1 <- predict(rpartFit, newdata = newsPopTest)

# compare the prediction vs the actual
resample1 <- postResample(pred1, obs = newsPopTest$shares)
resample1
```

    ##      RMSE  Rsquared       MAE 
    ## 0.4108117        NA 0.2030321

### Boosted Tree Based Model

A boosted tree is an ensemble method which slowly approaches the tree
prediction which would result from the original data. In general, an
ensemble model model will have a lower RSME than a single tree model.

``` r
gbmFit <- train(shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens
                 + num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + num_keywords + data_channel_is_lifestyle +
                 data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + data_channel_is_world +
                 self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity
                 + global_rate_positive_words + global_rate_negative_words + rate_positive_words + rate_negative_words + avg_positive_polarity +
                  min_positive_polarity + max_positive_polarity + avg_negative_polarity + min_negative_polarity + max_negative_polarity + title_subjectivity
                 + title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity, data = newsPopTrain,
             method = "gbm",
             trControl = trainControl(method = "cv", number = 10))
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.1884             nan     0.1000    0.0002
    ##      2        0.1880             nan     0.1000    0.0003
    ##      3        0.1872             nan     0.1000    0.0003
    ##      4        0.1863             nan     0.1000   -0.0001
    ##      5        0.1855             nan     0.1000    0.0001
    ##      6        0.1846             nan     0.1000   -0.0003
    ##      7        0.1840             nan     0.1000   -0.0003
    ##      8        0.1834             nan     0.1000    0.0000
    ##      9        0.1832             nan     0.1000    0.0001
    ##     10        0.1827             nan     0.1000   -0.0004
    ##     20        0.1795             nan     0.1000    0.0002
    ##     40        0.1746             nan     0.1000   -0.0004
    ##     60        0.1709             nan     0.1000    0.0001
    ##     80        0.1679             nan     0.1000    0.0001
    ##    100        0.1658             nan     0.1000   -0.0007
    ##    120        0.1644             nan     0.1000   -0.0003
    ##    140        0.1617             nan     0.1000   -0.0004
    ##    150        0.1606             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.1873             nan     0.1000    0.0009
    ##      2        0.1854             nan     0.1000   -0.0002
    ##      3        0.1838             nan     0.1000   -0.0003
    ##      4        0.1831             nan     0.1000    0.0003
    ##      5        0.1817             nan     0.1000   -0.0009
    ##      6        0.1810             nan     0.1000    0.0004
    ##      7        0.1801             nan     0.1000   -0.0003
    ##      8        0.1794             nan     0.1000   -0.0001
    ##      9        0.1789             nan     0.1000   -0.0004
    ##     10        0.1783             nan     0.1000   -0.0007
    ##     20        0.1727             nan     0.1000   -0.0006
    ##     40        0.1625             nan     0.1000    0.0000
    ##     60        0.1559             nan     0.1000    0.0001
    ##     80        0.1488             nan     0.1000   -0.0002
    ##    100        0.1438             nan     0.1000   -0.0006
    ##    120        0.1407             nan     0.1000   -0.0003
    ##    140        0.1367             nan     0.1000   -0.0001
    ##    150        0.1343             nan     0.1000   -0.0002
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.1882             nan     0.1000    0.0002
    ##      2        0.1868             nan     0.1000    0.0008
    ##      3        0.1847             nan     0.1000   -0.0002
    ##      4        0.1837             nan     0.1000   -0.0004
    ##      5        0.1830             nan     0.1000    0.0001
    ##      6        0.1820             nan     0.1000    0.0001
    ##      7        0.1806             nan     0.1000   -0.0000
    ##      8        0.1794             nan     0.1000   -0.0003
    ##      9        0.1787             nan     0.1000   -0.0003
    ##     10        0.1778             nan     0.1000   -0.0003
    ##     20        0.1693             nan     0.1000   -0.0007
    ##     40        0.1576             nan     0.1000   -0.0004
    ##     60        0.1487             nan     0.1000   -0.0005
    ##     80        0.1417             nan     0.1000   -0.0007
    ##    100        0.1321             nan     0.1000   -0.0006
    ##    120        0.1254             nan     0.1000   -0.0007
    ##    140        0.1186             nan     0.1000   -0.0005
    ##    150        0.1159             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0799             nan     0.1000   -0.0002
    ##      2        1.0793             nan     0.1000   -0.0005
    ##      3        1.0724             nan     0.1000    0.0002
    ##      4        1.0718             nan     0.1000   -0.0005
    ##      5        1.0677             nan     0.1000   -0.0008
    ##      6        1.0650             nan     0.1000   -0.0013
    ##      7        1.0628             nan     0.1000   -0.0008
    ##      8        1.0622             nan     0.1000    0.0004
    ##      9        1.0613             nan     0.1000   -0.0006
    ##     10        1.0607             nan     0.1000   -0.0002
    ##     20        1.0485             nan     0.1000   -0.0029
    ##     40        1.0293             nan     0.1000   -0.0017
    ##     60        1.0194             nan     0.1000   -0.0038
    ##     80        1.0119             nan     0.1000   -0.0028
    ##    100        1.0037             nan     0.1000    0.0011
    ##    120        0.9911             nan     0.1000   -0.0012
    ##    140        0.9864             nan     0.1000   -0.0029
    ##    150        0.9803             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0631             nan     0.1000   -0.0005
    ##      2        1.0459             nan     0.1000   -0.0004
    ##      3        1.0298             nan     0.1000   -0.0039
    ##      4        1.0291             nan     0.1000   -0.0006
    ##      5        1.0145             nan     0.1000   -0.0028
    ##      6        0.9977             nan     0.1000   -0.0016
    ##      7        0.9966             nan     0.1000    0.0002
    ##      8        0.9821             nan     0.1000   -0.0025
    ##      9        0.9683             nan     0.1000   -0.0016
    ##     10        0.9680             nan     0.1000   -0.0023
    ##     20        0.9159             nan     0.1000   -0.0048
    ##     40        0.8069             nan     0.1000   -0.0034
    ##     60        0.7544             nan     0.1000   -0.0004
    ##     80        0.6851             nan     0.1000   -0.0051
    ##    100        0.6507             nan     0.1000   -0.0060
    ##    120        0.6071             nan     0.1000   -0.0006
    ##    140        0.5747             nan     0.1000   -0.0028
    ##    150        0.5541             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0633             nan     0.1000   -0.0026
    ##      2        1.0480             nan     0.1000   -0.0029
    ##      3        1.0468             nan     0.1000   -0.0001
    ##      4        1.0290             nan     0.1000   -0.0019
    ##      5        1.0109             nan     0.1000   -0.0045
    ##      6        0.9928             nan     0.1000   -0.0007
    ##      7        0.9792             nan     0.1000   -0.0060
    ##      8        0.9650             nan     0.1000   -0.0024
    ##      9        0.9524             nan     0.1000   -0.0064
    ##     10        0.9411             nan     0.1000   -0.0060
    ##     20        0.8877             nan     0.1000   -0.0106
    ##     40        0.8393             nan     0.1000   -0.0098
    ##     60        0.7613             nan     0.1000   -0.0002
    ##     80        0.6840             nan     0.1000   -0.0051
    ##    100        0.6162             nan     0.1000   -0.0046
    ##    120        0.5615             nan     0.1000   -0.0065
    ##    140        0.5009             nan     0.1000   -0.0056
    ##    150        0.4822             nan     0.1000   -0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0954             nan     0.1000   -0.0005
    ##      2        1.0944             nan     0.1000   -0.0001
    ##      3        1.0910             nan     0.1000   -0.0010
    ##      4        1.0903             nan     0.1000   -0.0006
    ##      5        1.0871             nan     0.1000   -0.0001
    ##      6        1.0864             nan     0.1000   -0.0001
    ##      7        1.0860             nan     0.1000   -0.0004
    ##      8        1.0855             nan     0.1000   -0.0006
    ##      9        1.0831             nan     0.1000   -0.0009
    ##     10        1.0825             nan     0.1000   -0.0002
    ##     20        1.0672             nan     0.1000    0.0014
    ##     40        1.0512             nan     0.1000   -0.0023
    ##     60        1.0375             nan     0.1000   -0.0020
    ##     80        1.0301             nan     0.1000   -0.0035
    ##    100        1.0248             nan     0.1000   -0.0022
    ##    120        1.0172             nan     0.1000   -0.0029
    ##    140        1.0096             nan     0.1000   -0.0011
    ##    150        1.0080             nan     0.1000   -0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0840             nan     0.1000   -0.0002
    ##      2        1.0679             nan     0.1000   -0.0022
    ##      3        1.0662             nan     0.1000    0.0001
    ##      4        1.0506             nan     0.1000   -0.0040
    ##      5        1.0497             nan     0.1000   -0.0005
    ##      6        1.0486             nan     0.1000   -0.0000
    ##      7        1.0475             nan     0.1000    0.0002
    ##      8        1.0464             nan     0.1000   -0.0004
    ##      9        1.0456             nan     0.1000    0.0005
    ##     10        1.0438             nan     0.1000   -0.0006
    ##     20        0.9585             nan     0.1000    0.0005
    ##     40        0.8623             nan     0.1000   -0.0001
    ##     60        0.7883             nan     0.1000   -0.0050
    ##     80        0.7284             nan     0.1000   -0.0071
    ##    100        0.6680             nan     0.1000   -0.0055
    ##    120        0.6329             nan     0.1000   -0.0010
    ##    140        0.6050             nan     0.1000   -0.0034
    ##    150        0.5902             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1001             nan     0.1000   -0.0003
    ##      2        1.0828             nan     0.1000   -0.0020
    ##      3        1.0651             nan     0.1000   -0.0025
    ##      4        1.0640             nan     0.1000   -0.0007
    ##      5        1.0625             nan     0.1000    0.0001
    ##      6        1.0433             nan     0.1000    0.0002
    ##      7        1.0257             nan     0.1000   -0.0004
    ##      8        1.0125             nan     0.1000   -0.0037
    ##      9        0.9989             nan     0.1000   -0.0017
    ##     10        0.9841             nan     0.1000   -0.0029
    ##     20        0.9266             nan     0.1000   -0.0039
    ##     40        0.7989             nan     0.1000   -0.0018
    ##     60        0.6829             nan     0.1000   -0.0031
    ##     80        0.6089             nan     0.1000   -0.0121
    ##    100        0.5951             nan     0.1000   -0.0049
    ##    120        0.5395             nan     0.1000   -0.0019
    ##    140        0.4959             nan     0.1000   -0.0022
    ##    150        0.4792             nan     0.1000   -0.0040
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0996             nan     0.1000   -0.0001
    ##      2        1.0957             nan     0.1000   -0.0008
    ##      3        1.0946             nan     0.1000    0.0003
    ##      4        1.0919             nan     0.1000    0.0000
    ##      5        1.0892             nan     0.1000   -0.0007
    ##      6        1.0886             nan     0.1000   -0.0002
    ##      7        1.0877             nan     0.1000   -0.0001
    ##      8        1.0873             nan     0.1000   -0.0003
    ##      9        1.0873             nan     0.1000   -0.0007
    ##     10        1.0868             nan     0.1000    0.0002
    ##     20        1.0760             nan     0.1000   -0.0023
    ##     40        1.0597             nan     0.1000   -0.0010
    ##     60        1.0484             nan     0.1000   -0.0009
    ##     80        1.0399             nan     0.1000   -0.0025
    ##    100        1.0319             nan     0.1000   -0.0006
    ##    120        1.0220             nan     0.1000    0.0011
    ##    140        1.0175             nan     0.1000   -0.0034
    ##    150        1.0102             nan     0.1000    0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1023             nan     0.1000   -0.0007
    ##      2        1.0850             nan     0.1000   -0.0022
    ##      3        1.0611             nan     0.1000   -0.0049
    ##      4        1.0428             nan     0.1000   -0.0027
    ##      5        1.0411             nan     0.1000    0.0006
    ##      6        1.0393             nan     0.1000   -0.0005
    ##      7        1.0210             nan     0.1000   -0.0022
    ##      8        1.0072             nan     0.1000   -0.0028
    ##      9        1.0059             nan     0.1000    0.0002
    ##     10        0.9942             nan     0.1000   -0.0040
    ##     20        0.9803             nan     0.1000   -0.0026
    ##     40        0.8671             nan     0.1000   -0.0049
    ##     60        0.7846             nan     0.1000   -0.0034
    ##     80        0.7373             nan     0.1000   -0.0025
    ##    100        0.6742             nan     0.1000   -0.0024
    ##    120        0.6424             nan     0.1000   -0.0064
    ##    140        0.5887             nan     0.1000   -0.0002
    ##    150        0.5774             nan     0.1000   -0.0030
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1031             nan     0.1000   -0.0001
    ##      2        1.1025             nan     0.1000   -0.0001
    ##      3        1.0849             nan     0.1000   -0.0018
    ##      4        1.0637             nan     0.1000   -0.0025
    ##      5        1.0460             nan     0.1000   -0.0023
    ##      6        1.0445             nan     0.1000   -0.0010
    ##      7        1.0440             nan     0.1000   -0.0009
    ##      8        1.0419             nan     0.1000    0.0008
    ##      9        1.0285             nan     0.1000   -0.0027
    ##     10        1.0109             nan     0.1000    0.0007
    ##     20        0.9530             nan     0.1000   -0.0044
    ##     40        0.8347             nan     0.1000   -0.0027
    ##     60        0.7166             nan     0.1000   -0.0043
    ##     80        0.6638             nan     0.1000   -0.0049
    ##    100        0.6159             nan     0.1000   -0.0008
    ##    120        0.5753             nan     0.1000    0.0002
    ##    140        0.5328             nan     0.1000   -0.0096
    ##    150        0.5185             nan     0.1000   -0.0055
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0926             nan     0.1000   -0.0004
    ##      2        1.0918             nan     0.1000   -0.0003
    ##      3        1.0916             nan     0.1000   -0.0005
    ##      4        1.0908             nan     0.1000   -0.0002
    ##      5        1.0901             nan     0.1000   -0.0002
    ##      6        1.0845             nan     0.1000    0.0013
    ##      7        1.0841             nan     0.1000   -0.0005
    ##      8        1.0800             nan     0.1000   -0.0010
    ##      9        1.0772             nan     0.1000   -0.0007
    ##     10        1.0767             nan     0.1000   -0.0005
    ##     20        1.0665             nan     0.1000   -0.0007
    ##     40        1.0545             nan     0.1000   -0.0004
    ##     60        1.0402             nan     0.1000    0.0010
    ##     80        1.0279             nan     0.1000   -0.0004
    ##    100        1.0190             nan     0.1000   -0.0005
    ##    120        1.0120             nan     0.1000    0.0012
    ##    140        1.0070             nan     0.1000    0.0009
    ##    150        1.0048             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0929             nan     0.1000   -0.0005
    ##      2        1.0917             nan     0.1000    0.0006
    ##      3        1.0716             nan     0.1000   -0.0007
    ##      4        1.0535             nan     0.1000   -0.0084
    ##      5        1.0372             nan     0.1000    0.0002
    ##      6        1.0182             nan     0.1000    0.0038
    ##      7        1.0169             nan     0.1000    0.0001
    ##      8        1.0166             nan     0.1000   -0.0005
    ##      9        1.0014             nan     0.1000   -0.0086
    ##     10        0.9879             nan     0.1000   -0.0030
    ##     20        0.9314             nan     0.1000   -0.0004
    ##     40        0.8304             nan     0.1000   -0.0041
    ##     60        0.7599             nan     0.1000   -0.0012
    ##     80        0.6970             nan     0.1000   -0.0036
    ##    100        0.6589             nan     0.1000   -0.0056
    ##    120        0.5972             nan     0.1000   -0.0057
    ##    140        0.5651             nan     0.1000    0.0013
    ##    150        0.5512             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0745             nan     0.1000   -0.0015
    ##      2        1.0725             nan     0.1000    0.0009
    ##      3        1.0568             nan     0.1000   -0.0022
    ##      4        1.0417             nan     0.1000   -0.0034
    ##      5        1.0239             nan     0.1000   -0.0030
    ##      6        1.0222             nan     0.1000    0.0006
    ##      7        1.0047             nan     0.1000   -0.0007
    ##      8        0.9909             nan     0.1000   -0.0015
    ##      9        0.9792             nan     0.1000   -0.0056
    ##     10        0.9775             nan     0.1000   -0.0010
    ##     20        0.8874             nan     0.1000   -0.0014
    ##     40        0.7880             nan     0.1000   -0.0017
    ##     60        0.6787             nan     0.1000   -0.0038
    ##     80        0.6486             nan     0.1000   -0.0069
    ##    100        0.5904             nan     0.1000   -0.0038
    ##    120        0.5467             nan     0.1000   -0.0001
    ##    140        0.5219             nan     0.1000   -0.0008
    ##    150        0.4969             nan     0.1000   -0.0066
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.1001             nan     0.1000   -0.0002
    ##      2        1.0947             nan     0.1000    0.0002
    ##      3        1.0914             nan     0.1000   -0.0007
    ##      4        1.0909             nan     0.1000    0.0001
    ##      5        1.0900             nan     0.1000   -0.0008
    ##      6        1.0868             nan     0.1000    0.0009
    ##      7        1.0862             nan     0.1000   -0.0005
    ##      8        1.0836             nan     0.1000    0.0002
    ##      9        1.0833             nan     0.1000   -0.0004
    ##     10        1.0828             nan     0.1000    0.0005
    ##     20        1.0736             nan     0.1000   -0.0003
    ##     40        1.0473             nan     0.1000   -0.0008
    ##     60        1.0396             nan     0.1000   -0.0022
    ##     80        1.0336             nan     0.1000    0.0008
    ##    100        1.0249             nan     0.1000    0.0011
    ##    120        1.0190             nan     0.1000   -0.0010
    ##    140        1.0109             nan     0.1000    0.0004
    ##    150        1.0068             nan     0.1000   -0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0761             nan     0.1000   -0.0004
    ##      2        1.0593             nan     0.1000   -0.0014
    ##      3        1.0573             nan     0.1000   -0.0001
    ##      4        1.0560             nan     0.1000    0.0000
    ##      5        1.0399             nan     0.1000   -0.0016
    ##      6        1.0226             nan     0.1000   -0.0016
    ##      7        1.0219             nan     0.1000   -0.0006
    ##      8        1.0095             nan     0.1000   -0.0073
    ##      9        0.9926             nan     0.1000   -0.0015
    ##     10        0.9915             nan     0.1000    0.0002
    ##     20        0.9615             nan     0.1000   -0.0026
    ##     40        0.8615             nan     0.1000   -0.0004
    ##     60        0.7620             nan     0.1000   -0.0007
    ##     80        0.7031             nan     0.1000   -0.0037
    ##    100        0.6634             nan     0.1000   -0.0022
    ##    120        0.6185             nan     0.1000   -0.0018
    ##    140        0.5847             nan     0.1000   -0.0004
    ##    150        0.5723             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0994             nan     0.1000   -0.0003
    ##      2        1.0778             nan     0.1000   -0.0004
    ##      3        1.0771             nan     0.1000   -0.0003
    ##      4        1.0751             nan     0.1000   -0.0006
    ##      5        1.0739             nan     0.1000   -0.0007
    ##      6        1.0730             nan     0.1000   -0.0004
    ##      7        1.0480             nan     0.1000   -0.0049
    ##      8        1.0279             nan     0.1000   -0.0012
    ##      9        1.0266             nan     0.1000   -0.0009
    ##     10        1.0255             nan     0.1000   -0.0004
    ##     20        0.9738             nan     0.1000   -0.0059
    ##     40        0.8296             nan     0.1000   -0.0027
    ##     60        0.7427             nan     0.1000   -0.0022
    ##     80        0.6961             nan     0.1000   -0.0087
    ##    100        0.6396             nan     0.1000   -0.0056
    ##    120        0.5756             nan     0.1000    0.0004
    ##    140        0.5360             nan     0.1000   -0.0030
    ##    150        0.5184             nan     0.1000   -0.0039
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0928             nan     0.1000   -0.0000
    ##      2        1.0920             nan     0.1000    0.0009
    ##      3        1.0885             nan     0.1000    0.0009
    ##      4        1.0880             nan     0.1000   -0.0005
    ##      5        1.0854             nan     0.1000    0.0002
    ##      6        1.0833             nan     0.1000   -0.0005
    ##      7        1.0805             nan     0.1000    0.0008
    ##      8        1.0797             nan     0.1000    0.0002
    ##      9        1.0780             nan     0.1000   -0.0002
    ##     10        1.0774             nan     0.1000   -0.0008
    ##     20        1.0686             nan     0.1000   -0.0027
    ##     40        1.0553             nan     0.1000   -0.0011
    ##     60        1.0428             nan     0.1000   -0.0013
    ##     80        1.0362             nan     0.1000   -0.0021
    ##    100        1.0295             nan     0.1000   -0.0035
    ##    120        1.0197             nan     0.1000   -0.0014
    ##    140        1.0113             nan     0.1000   -0.0018
    ##    150        1.0085             nan     0.1000   -0.0042
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0977             nan     0.1000   -0.0004
    ##      2        1.0801             nan     0.1000   -0.0025
    ##      3        1.0612             nan     0.1000   -0.0079
    ##      4        1.0603             nan     0.1000    0.0001
    ##      5        1.0598             nan     0.1000   -0.0004
    ##      6        1.0585             nan     0.1000   -0.0006
    ##      7        1.0571             nan     0.1000   -0.0001
    ##      8        1.0567             nan     0.1000   -0.0006
    ##      9        1.0559             nan     0.1000   -0.0004
    ##     10        1.0394             nan     0.1000   -0.0030
    ##     20        0.9557             nan     0.1000   -0.0018
    ##     40        0.8515             nan     0.1000   -0.0078
    ##     60        0.7668             nan     0.1000   -0.0018
    ##     80        0.6985             nan     0.1000   -0.0055
    ##    100        0.6582             nan     0.1000   -0.0006
    ##    120        0.6150             nan     0.1000   -0.0031
    ##    140        0.5708             nan     0.1000   -0.0011
    ##    150        0.5510             nan     0.1000   -0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0801             nan     0.1000   -0.0008
    ##      2        1.0605             nan     0.1000   -0.0018
    ##      3        1.0466             nan     0.1000   -0.0054
    ##      4        1.0453             nan     0.1000    0.0003
    ##      5        1.0442             nan     0.1000    0.0002
    ##      6        1.0435             nan     0.1000   -0.0026
    ##      7        1.0256             nan     0.1000   -0.0053
    ##      8        1.0271             nan     0.1000   -0.0060
    ##      9        1.0137             nan     0.1000   -0.0040
    ##     10        1.0132             nan     0.1000   -0.0020
    ##     20        0.9698             nan     0.1000   -0.0077
    ##     40        0.8581             nan     0.1000   -0.0013
    ##     60        0.7847             nan     0.1000    0.0007
    ##     80        0.7130             nan     0.1000   -0.0052
    ##    100        0.6662             nan     0.1000   -0.0040
    ##    120        0.6150             nan     0.1000   -0.0037
    ##    140        0.5526             nan     0.1000   -0.0031
    ##    150        0.5365             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0804             nan     0.1000   -0.0004
    ##      2        1.0797             nan     0.1000    0.0001
    ##      3        1.0757             nan     0.1000   -0.0017
    ##      4        1.0750             nan     0.1000   -0.0003
    ##      5        1.0743             nan     0.1000   -0.0009
    ##      6        1.0717             nan     0.1000   -0.0013
    ##      7        1.0687             nan     0.1000    0.0002
    ##      8        1.0664             nan     0.1000   -0.0002
    ##      9        1.0660             nan     0.1000   -0.0006
    ##     10        1.0659             nan     0.1000   -0.0001
    ##     20        1.0559             nan     0.1000   -0.0004
    ##     40        1.0393             nan     0.1000   -0.0028
    ##     60        1.0262             nan     0.1000   -0.0028
    ##     80        1.0192             nan     0.1000   -0.0015
    ##    100        1.0104             nan     0.1000   -0.0026
    ##    120        1.0017             nan     0.1000   -0.0020
    ##    140        0.9885             nan     0.1000   -0.0015
    ##    150        0.9888             nan     0.1000   -0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0850             nan     0.1000   -0.0004
    ##      2        1.0845             nan     0.1000   -0.0002
    ##      3        1.0834             nan     0.1000   -0.0006
    ##      4        1.0664             nan     0.1000   -0.0013
    ##      5        1.0658             nan     0.1000   -0.0005
    ##      6        1.0482             nan     0.1000   -0.0008
    ##      7        1.0479             nan     0.1000   -0.0002
    ##      8        1.0473             nan     0.1000   -0.0012
    ##      9        1.0468             nan     0.1000   -0.0005
    ##     10        1.0289             nan     0.1000   -0.0106
    ##     20        0.9259             nan     0.1000   -0.0043
    ##     40        0.8521             nan     0.1000   -0.0090
    ##     60        0.8032             nan     0.1000   -0.0081
    ##     80        0.7039             nan     0.1000   -0.0028
    ##    100        0.6495             nan     0.1000   -0.0006
    ##    120        0.6150             nan     0.1000   -0.0021
    ##    140        0.5775             nan     0.1000   -0.0009
    ##    150        0.5637             nan     0.1000   -0.0053
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0680             nan     0.1000   -0.0031
    ##      2        1.0667             nan     0.1000   -0.0008
    ##      3        1.0512             nan     0.1000   -0.0027
    ##      4        1.0491             nan     0.1000   -0.0001
    ##      5        1.0482             nan     0.1000   -0.0009
    ##      6        1.0335             nan     0.1000   -0.0059
    ##      7        1.0353             nan     0.1000   -0.0054
    ##      8        1.0149             nan     0.1000   -0.0039
    ##      9        0.9976             nan     0.1000   -0.0023
    ##     10        0.9814             nan     0.1000   -0.0030
    ##     20        0.9270             nan     0.1000   -0.0037
    ##     40        0.8103             nan     0.1000   -0.0076
    ##     60        0.7120             nan     0.1000   -0.0078
    ##     80        0.6300             nan     0.1000   -0.0087
    ##    100        0.5707             nan     0.1000   -0.0024
    ##    120        0.5049             nan     0.1000   -0.0036
    ##    140        0.4510             nan     0.1000   -0.0001
    ##    150        0.4443             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0428             nan     0.1000    0.0001
    ##      2        1.0426             nan     0.1000   -0.0004
    ##      3        1.0395             nan     0.1000   -0.0004
    ##      4        1.0372             nan     0.1000   -0.0019
    ##      5        1.0365             nan     0.1000    0.0001
    ##      6        1.0357             nan     0.1000    0.0001
    ##      7        1.0351             nan     0.1000   -0.0000
    ##      8        1.0349             nan     0.1000   -0.0001
    ##      9        1.0353             nan     0.1000   -0.0015
    ##     10        1.0353             nan     0.1000   -0.0004
    ##     20        1.0268             nan     0.1000    0.0002
    ##     40        1.0122             nan     0.1000   -0.0009
    ##     60        1.0039             nan     0.1000   -0.0013
    ##     80        0.9963             nan     0.1000   -0.0019
    ##    100        0.9910             nan     0.1000   -0.0019
    ##    120        0.9891             nan     0.1000   -0.0017
    ##    140        0.9860             nan     0.1000   -0.0013
    ##    150        0.9858             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0293             nan     0.1000   -0.0010
    ##      2        1.0135             nan     0.1000   -0.0020
    ##      3        0.9994             nan     0.1000   -0.0022
    ##      4        0.9986             nan     0.1000   -0.0000
    ##      5        0.9848             nan     0.1000   -0.0045
    ##      6        0.9661             nan     0.1000   -0.0019
    ##      7        0.9655             nan     0.1000    0.0004
    ##      8        0.9499             nan     0.1000   -0.0040
    ##      9        0.9373             nan     0.1000   -0.0040
    ##     10        0.9361             nan     0.1000    0.0002
    ##     20        0.8755             nan     0.1000   -0.0011
    ##     40        0.7524             nan     0.1000   -0.0006
    ##     60        0.7059             nan     0.1000   -0.0049
    ##     80        0.6621             nan     0.1000   -0.0019
    ##    100        0.6254             nan     0.1000   -0.0004
    ##    120        0.5698             nan     0.1000   -0.0031
    ##    140        0.5326             nan     0.1000   -0.0001
    ##    150        0.5230             nan     0.1000   -0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0446             nan     0.1000    0.0010
    ##      2        1.0274             nan     0.1000   -0.0012
    ##      3        1.0109             nan     0.1000   -0.0011
    ##      4        0.9955             nan     0.1000   -0.0029
    ##      5        0.9965             nan     0.1000   -0.0047
    ##      6        0.9828             nan     0.1000   -0.0046
    ##      7        0.9688             nan     0.1000   -0.0046
    ##      8        0.9694             nan     0.1000   -0.0038
    ##      9        0.9688             nan     0.1000   -0.0015
    ##     10        0.9702             nan     0.1000   -0.0055
    ##     20        0.8710             nan     0.1000   -0.0015
    ##     40        0.7636             nan     0.1000   -0.0012
    ##     60        0.7051             nan     0.1000   -0.0040
    ##     80        0.6471             nan     0.1000   -0.0074
    ##    100        0.5877             nan     0.1000    0.0017
    ##    120        0.5397             nan     0.1000   -0.0040
    ##    140        0.5101             nan     0.1000   -0.0047
    ##    150        0.4847             nan     0.1000   -0.0044
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0915             nan     0.1000    0.0002
    ##      2        1.0863             nan     0.1000   -0.0007
    ##      3        1.0860             nan     0.1000   -0.0002
    ##      4        1.0823             nan     0.1000    0.0001
    ##      5        1.0798             nan     0.1000   -0.0011
    ##      6        1.0762             nan     0.1000    0.0000
    ##      7        1.0739             nan     0.1000   -0.0001
    ##      8        1.0734             nan     0.1000   -0.0003
    ##      9        1.0711             nan     0.1000   -0.0014
    ##     10        1.0696             nan     0.1000   -0.0007
    ##     20        1.0618             nan     0.1000   -0.0024
    ##     40        1.0425             nan     0.1000   -0.0014
    ##     60        1.0271             nan     0.1000    0.0010
    ##     80        1.0168             nan     0.1000   -0.0018
    ##    100        1.0089             nan     0.1000   -0.0018
    ##    120        1.0011             nan     0.1000    0.0012
    ##    140        0.9969             nan     0.1000   -0.0026
    ##    150        0.9931             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0686             nan     0.1000   -0.0027
    ##      2        1.0673             nan     0.1000    0.0007
    ##      3        1.0663             nan     0.1000   -0.0006
    ##      4        1.0649             nan     0.1000    0.0005
    ##      5        1.0476             nan     0.1000   -0.0025
    ##      6        1.0296             nan     0.1000   -0.0033
    ##      7        1.0290             nan     0.1000   -0.0004
    ##      8        1.0143             nan     0.1000   -0.0104
    ##      9        0.9984             nan     0.1000   -0.0013
    ##     10        0.9975             nan     0.1000   -0.0003
    ##     20        0.8982             nan     0.1000   -0.0086
    ##     40        0.8000             nan     0.1000   -0.0084
    ##     60        0.7703             nan     0.1000   -0.0053
    ##     80        0.7215             nan     0.1000   -0.0092
    ##    100        0.6617             nan     0.1000    0.0006
    ##    120        0.5974             nan     0.1000   -0.0041
    ##    140        0.5540             nan     0.1000   -0.0009
    ##    150        0.5414             nan     0.1000   -0.0001
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0917             nan     0.1000    0.0001
    ##      2        1.0895             nan     0.1000    0.0008
    ##      3        1.0881             nan     0.1000    0.0002
    ##      4        1.0871             nan     0.1000    0.0003
    ##      5        1.0646             nan     0.1000   -0.0034
    ##      6        1.0634             nan     0.1000    0.0004
    ##      7        1.0628             nan     0.1000   -0.0005
    ##      8        1.0616             nan     0.1000   -0.0007
    ##      9        1.0604             nan     0.1000    0.0002
    ##     10        1.0597             nan     0.1000    0.0000
    ##     20        0.9801             nan     0.1000   -0.0044
    ##     40        0.8388             nan     0.1000   -0.0052
    ##     60        0.7542             nan     0.1000   -0.0060
    ##     80        0.6793             nan     0.1000   -0.0064
    ##    100        0.6212             nan     0.1000   -0.0076
    ##    120        0.5600             nan     0.1000   -0.0001
    ##    140        0.5031             nan     0.1000   -0.0006
    ##    150        0.4771             nan     0.1000   -0.0057
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9985             nan     0.1000   -0.0003
    ##      2        0.9980             nan     0.1000   -0.0002
    ##      3        0.9975             nan     0.1000   -0.0003
    ##      4        0.9975             nan     0.1000   -0.0005
    ##      5        0.9940             nan     0.1000    0.0015
    ##      6        0.9912             nan     0.1000    0.0000
    ##      7        0.9908             nan     0.1000   -0.0004
    ##      8        0.9877             nan     0.1000    0.0009
    ##      9        0.9875             nan     0.1000   -0.0001
    ##     10        0.9872             nan     0.1000   -0.0002
    ##     20        0.9758             nan     0.1000    0.0010
    ##     40        0.9575             nan     0.1000   -0.0007
    ##     50        0.9533             nan     0.1000   -0.0004

``` r
gbmFit
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 1719 samples
    ##   37 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1548, 1547, 1547, 1546, 1546, 1549, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared     MAE      
    ##   1                   50      0.6694137  0.012363833  0.2254894
    ##   1                  100      0.6733203  0.014127744  0.2297206
    ##   1                  150      0.6802995  0.016749355  0.2365065
    ##   2                   50      0.7230530  0.006492740  0.2445872
    ##   2                  100      0.7625313  0.006790371  0.2668683
    ##   2                  150      0.7798891  0.011797078  0.2862874
    ##   3                   50      0.7236062  0.014307952  0.2478826
    ##   3                  100      0.7574517  0.008698497  0.2671671
    ##   3                  150      0.7881499  0.009471947  0.2878366
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
# create the prediction
pred2 <- predict(gbmFit, newdata = newsPopTest)

# compare the prediction vs the actual
resample2 <- postResample(pred2, obs = newsPopTest$shares)
resample2
```

    ##       RMSE   Rsquared        MAE 
    ## 0.42296448 0.01146643 0.21408427

### Comparison

Below is a comparison of the two methods. Both have relatively high root
mean square errors.

``` r
comparison <- data.frame("RSME" = c(resample1[[1]], resample2[[1]]), "MAE" = c(resample1[[3]], resample2[[3]]) )
rownames(comparison) <- c("RPART","GBM")
kable(comparison)
```

<table>

<thead>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:right;">

RSME

</th>

<th style="text-align:right;">

MAE

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

RPART

</td>

<td style="text-align:right;">

0.4108117

</td>

<td style="text-align:right;">

0.2030321

</td>

</tr>

<tr>

<td style="text-align:left;">

GBM

</td>

<td style="text-align:right;">

0.4229645

</td>

<td style="text-align:right;">

0.2140843

</td>

</tr>

</tbody>

</table>
