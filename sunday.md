News Popularity Sunday Data
================
Shuang Du
10/16/2020

Load Libraries
--------------

    library(readxl);
    library(tidyverse);
    library(caret);
    library(modelr);
    library(rpart);
    library(kableExtra);

Read in Data
------------

    getData <- function(day) {

      newsPopData <- read_csv("raw_data\\OnlineNewsPopularity.csv")
      
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

Set Aside Training Data
-----------------------

    set.seed(92)
    trainIndex <- createDataPartition(newsPopData$shares, 
                                      p = 0.7, list = FALSE)

    newsPopTrain <- newsPopData[as.vector(trainIndex),];
    newsPopTest <- newsPopData[-as.vector(trainIndex),];

Center and Scale
----------------

    preProcValues <- preProcess(newsPopTrain, method = c("center", "scale"))
    newsPopTrain <- predict(preProcValues, newsPopTrain) 
    newsPopTest <- predict(preProcValues, newsPopTest)

Summary of a Few Variables
--------------------------

The plots below show a histogram of the number of shares for the given
day. Scatter plots on the effect of max positive polarity, article time
delta and number of videos in the article are also included.

As expected the histogram has a strong right tail, as seem by the
summary stats which show a very high maximum and a median severals
orders of magnitude lower. This is expected for because of the “viral”
nature of online popularity.

    summary(newsPopTrain$shares)

    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -0.610731 -0.423911 -0.306203  0.000000 -0.003526 11.061014

    g0 <- ggplot(newsPopTrain, aes(x=shares))
    g0 + geom_histogram(binwidth = 0.5) + ggtitle('Histogram for Number of Shares') + ylab('Number of Shares') + xlab('Shares')

![](sunday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

    summary(newsPopTrain$max_positive_polarity)

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -3.1745 -0.7367  0.0759  0.0000  0.8885  0.8885

    g1 <- ggplot(newsPopTrain, aes(x = max_positive_polarity, y = shares )) 
    g1 + geom_point() + ggtitle('Scatter of Max Positive Polarity Effect') + ylab('Shares') + xlab('Max Positive Polarity')

![](sunday_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

    summary(newsPopTrain$timedelta)

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -1.6340 -0.8773 -0.0548  0.0000  0.8664  1.7218

    g2 <- ggplot(newsPopTrain, aes(x = timedelta, y = shares )) 
    g2 + geom_point() + ggtitle('Scatter of Article Age Effect') + ylab('Shares') + xlab('Time Delta')

![](sunday_files/figure-gfm/unnamed-chunk-5-3.png)<!-- -->

    summary(newsPopTrain$num_videos)

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -0.2929 -0.2929 -0.2929  0.0000  0.0000 21.0876

    g3 <- ggplot(newsPopTrain, aes(x = num_videos, y = shares )) 
    g3 + geom_point() + ggtitle('Scatter of Videos Number Effect') + ylab('Shares') + xlab('Number of Videos')

![](sunday_files/figure-gfm/unnamed-chunk-5-4.png)<!-- -->

Modeling
--------

### Standard Tree Based Model (no ensemble)

The type of model being fitted here is a decision tree. The tree splits
are based on minimizing the residual sum of squares for each region.

    rpartFit <- train(shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens
                     + num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + num_keywords + data_channel_is_lifestyle +
                     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + data_channel_is_world +
                     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity
                     + global_rate_positive_words + global_rate_negative_words + rate_positive_words + rate_negative_words + avg_positive_polarity +
                      min_positive_polarity + max_positive_polarity + avg_negative_polarity + min_negative_polarity + max_negative_polarity + title_subjectivity
                     + title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity, data = newsPopTrain,
                 method = "rpart",
                 trControl = trainControl(method = "cv", number = 10),
                 tuneGrid = data.frame(cp = c(.001,.01,.015,.02,.03,.04,.05))
                 )
    rpartFit

    ## CART 
    ## 
    ## 1917 samples
    ##   37 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1724, 1724, 1725, 1726, 1725, 1725, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp     RMSE       Rsquared      MAE      
    ##   0.001  1.1010316  0.0113300172  0.5597200
    ##   0.010  1.0408693  0.0126844835  0.5166058
    ##   0.015  1.0252275  0.0014898800  0.5180181
    ##   0.020  1.0185979  0.0005610921  0.5215812
    ##   0.030  0.9878788           NaN  0.5094454
    ##   0.040  0.9878788           NaN  0.5094454
    ##   0.050  0.9878788           NaN  0.5094454
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.05.

    # create the prediction
    pred1 <- predict(rpartFit, newdata = newsPopTest)

    # compare the prediction vs the actual
    resample1 <- postResample(pred1, obs = newsPopTest$shares)
    resample1

    ##      RMSE  Rsquared       MAE 
    ## 1.1434027        NA 0.5214574

### Boosted Tree Based Model

A boosted tree is an ensemble method which slowly approaches the tree
prediction which would result from the original data. In general, an
ensemble model model will have a lower RSME than a single tree model.

    gbmFit <- train(shares ~ timedelta + n_tokens_title + n_tokens_content + n_unique_tokens + n_non_stop_words + n_non_stop_unique_tokens
                     + num_hrefs + num_self_hrefs + num_imgs + num_videos + average_token_length + num_keywords + data_channel_is_lifestyle +
                     data_channel_is_entertainment + data_channel_is_bus + data_channel_is_socmed + data_channel_is_tech + data_channel_is_world +
                     self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + global_subjectivity + global_sentiment_polarity
                     + global_rate_positive_words + global_rate_negative_words + rate_positive_words + rate_negative_words + avg_positive_polarity +
                      min_positive_polarity + max_positive_polarity + avg_negative_polarity + min_negative_polarity + max_negative_polarity + title_subjectivity
                     + title_sentiment_polarity + abs_title_subjectivity + abs_title_sentiment_polarity, data = newsPopTrain,
                 method = "gbm",
                 trControl = trainControl(method = "cv", number = 10))

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9727             nan     0.1000   -0.0006
    ##      2        0.9689             nan     0.1000   -0.0007
    ##      3        0.9656             nan     0.1000    0.0037
    ##      4        0.9615             nan     0.1000    0.0029
    ##      5        0.9589             nan     0.1000    0.0016
    ##      6        0.9568             nan     0.1000   -0.0003
    ##      7        0.9539             nan     0.1000    0.0017
    ##      8        0.9522             nan     0.1000   -0.0011
    ##      9        0.9505             nan     0.1000   -0.0008
    ##     10        0.9486             nan     0.1000   -0.0003
    ##     20        0.9331             nan     0.1000   -0.0003
    ##     40        0.9207             nan     0.1000   -0.0009
    ##     60        0.9075             nan     0.1000   -0.0007
    ##     80        0.9003             nan     0.1000   -0.0024
    ##    100        0.8914             nan     0.1000   -0.0009
    ##    120        0.8843             nan     0.1000   -0.0023
    ##    140        0.8793             nan     0.1000   -0.0031
    ##    150        0.8751             nan     0.1000   -0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9708             nan     0.1000    0.0017
    ##      2        0.9678             nan     0.1000    0.0008
    ##      3        0.9623             nan     0.1000   -0.0021
    ##      4        0.9584             nan     0.1000   -0.0004
    ##      5        0.9542             nan     0.1000   -0.0017
    ##      6        0.9480             nan     0.1000    0.0033
    ##      7        0.9443             nan     0.1000   -0.0005
    ##      8        0.9386             nan     0.1000    0.0005
    ##      9        0.9355             nan     0.1000   -0.0005
    ##     10        0.9272             nan     0.1000   -0.0038
    ##     20        0.8938             nan     0.1000   -0.0017
    ##     40        0.8569             nan     0.1000   -0.0037
    ##     60        0.8260             nan     0.1000   -0.0030
    ##     80        0.8070             nan     0.1000   -0.0018
    ##    100        0.7878             nan     0.1000   -0.0022
    ##    120        0.7659             nan     0.1000   -0.0018
    ##    140        0.7422             nan     0.1000   -0.0012
    ##    150        0.7343             nan     0.1000   -0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9713             nan     0.1000    0.0014
    ##      2        0.9664             nan     0.1000   -0.0023
    ##      3        0.9591             nan     0.1000    0.0015
    ##      4        0.9538             nan     0.1000    0.0015
    ##      5        0.9479             nan     0.1000    0.0023
    ##      6        0.9382             nan     0.1000   -0.0005
    ##      7        0.9319             nan     0.1000    0.0017
    ##      8        0.9236             nan     0.1000    0.0033
    ##      9        0.9184             nan     0.1000   -0.0008
    ##     10        0.9118             nan     0.1000    0.0030
    ##     20        0.8748             nan     0.1000   -0.0026
    ##     40        0.8074             nan     0.1000   -0.0026
    ##     60        0.7625             nan     0.1000   -0.0021
    ##     80        0.7332             nan     0.1000   -0.0026
    ##    100        0.6956             nan     0.1000   -0.0011
    ##    120        0.6602             nan     0.1000   -0.0010
    ##    140        0.6359             nan     0.1000   -0.0025
    ##    150        0.6255             nan     0.1000   -0.0030
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9882             nan     0.1000   -0.0006
    ##      2        0.9852             nan     0.1000   -0.0015
    ##      3        0.9816             nan     0.1000    0.0027
    ##      4        0.9791             nan     0.1000    0.0024
    ##      5        0.9763             nan     0.1000    0.0013
    ##      6        0.9739             nan     0.1000   -0.0012
    ##      7        0.9714             nan     0.1000   -0.0007
    ##      8        0.9689             nan     0.1000   -0.0015
    ##      9        0.9673             nan     0.1000   -0.0013
    ##     10        0.9653             nan     0.1000    0.0005
    ##     20        0.9523             nan     0.1000   -0.0008
    ##     40        0.9360             nan     0.1000   -0.0018
    ##     60        0.9254             nan     0.1000   -0.0023
    ##     80        0.9188             nan     0.1000   -0.0011
    ##    100        0.9107             nan     0.1000   -0.0003
    ##    120        0.9050             nan     0.1000   -0.0025
    ##    140        0.8996             nan     0.1000   -0.0016
    ##    150        0.8980             nan     0.1000   -0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9863             nan     0.1000    0.0002
    ##      2        0.9816             nan     0.1000    0.0006
    ##      3        0.9771             nan     0.1000   -0.0005
    ##      4        0.9738             nan     0.1000   -0.0024
    ##      5        0.9707             nan     0.1000    0.0023
    ##      6        0.9666             nan     0.1000   -0.0001
    ##      7        0.9626             nan     0.1000    0.0012
    ##      8        0.9599             nan     0.1000   -0.0024
    ##      9        0.9561             nan     0.1000    0.0001
    ##     10        0.9532             nan     0.1000   -0.0017
    ##     20        0.9222             nan     0.1000   -0.0019
    ##     40        0.8811             nan     0.1000   -0.0013
    ##     60        0.8568             nan     0.1000   -0.0017
    ##     80        0.8321             nan     0.1000   -0.0023
    ##    100        0.8078             nan     0.1000   -0.0017
    ##    120        0.7927             nan     0.1000   -0.0008
    ##    140        0.7749             nan     0.1000   -0.0012
    ##    150        0.7636             nan     0.1000   -0.0003
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9829             nan     0.1000    0.0006
    ##      2        0.9721             nan     0.1000    0.0007
    ##      3        0.9664             nan     0.1000   -0.0016
    ##      4        0.9564             nan     0.1000    0.0035
    ##      5        0.9509             nan     0.1000   -0.0008
    ##      6        0.9440             nan     0.1000   -0.0008
    ##      7        0.9389             nan     0.1000    0.0001
    ##      8        0.9345             nan     0.1000    0.0006
    ##      9        0.9307             nan     0.1000   -0.0009
    ##     10        0.9212             nan     0.1000   -0.0077
    ##     20        0.8823             nan     0.1000   -0.0020
    ##     40        0.8348             nan     0.1000   -0.0019
    ##     60        0.7872             nan     0.1000   -0.0029
    ##     80        0.7495             nan     0.1000   -0.0016
    ##    100        0.7217             nan     0.1000   -0.0015
    ##    120        0.6989             nan     0.1000   -0.0014
    ##    140        0.6688             nan     0.1000   -0.0023
    ##    150        0.6505             nan     0.1000   -0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0003             nan     0.1000    0.0017
    ##      2        0.9977             nan     0.1000   -0.0011
    ##      3        0.9951             nan     0.1000    0.0017
    ##      4        0.9917             nan     0.1000   -0.0002
    ##      5        0.9900             nan     0.1000   -0.0004
    ##      6        0.9879             nan     0.1000    0.0002
    ##      7        0.9854             nan     0.1000    0.0009
    ##      8        0.9837             nan     0.1000    0.0000
    ##      9        0.9808             nan     0.1000   -0.0025
    ##     10        0.9802             nan     0.1000   -0.0011
    ##     20        0.9666             nan     0.1000   -0.0012
    ##     40        0.9507             nan     0.1000   -0.0007
    ##     60        0.9404             nan     0.1000   -0.0004
    ##     80        0.9333             nan     0.1000   -0.0013
    ##    100        0.9247             nan     0.1000   -0.0025
    ##    120        0.9182             nan     0.1000   -0.0023
    ##    140        0.9113             nan     0.1000   -0.0014
    ##    150        0.9090             nan     0.1000   -0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9967             nan     0.1000   -0.0007
    ##      2        0.9908             nan     0.1000   -0.0016
    ##      3        0.9862             nan     0.1000    0.0015
    ##      4        0.9833             nan     0.1000   -0.0006
    ##      5        0.9764             nan     0.1000   -0.0031
    ##      6        0.9714             nan     0.1000   -0.0029
    ##      7        0.9675             nan     0.1000    0.0009
    ##      8        0.9645             nan     0.1000    0.0006
    ##      9        0.9592             nan     0.1000    0.0003
    ##     10        0.9556             nan     0.1000   -0.0016
    ##     20        0.9300             nan     0.1000   -0.0023
    ##     40        0.8925             nan     0.1000   -0.0006
    ##     60        0.8661             nan     0.1000   -0.0019
    ##     80        0.8443             nan     0.1000   -0.0019
    ##    100        0.8201             nan     0.1000   -0.0041
    ##    120        0.8006             nan     0.1000   -0.0015
    ##    140        0.7771             nan     0.1000   -0.0015
    ##    150        0.7657             nan     0.1000   -0.0005
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9976             nan     0.1000   -0.0006
    ##      2        0.9942             nan     0.1000    0.0004
    ##      3        0.9878             nan     0.1000   -0.0010
    ##      4        0.9782             nan     0.1000   -0.0015
    ##      5        0.9693             nan     0.1000   -0.0011
    ##      6        0.9638             nan     0.1000   -0.0005
    ##      7        0.9611             nan     0.1000    0.0004
    ##      8        0.9571             nan     0.1000    0.0006
    ##      9        0.9505             nan     0.1000    0.0004
    ##     10        0.9446             nan     0.1000   -0.0016
    ##     20        0.8927             nan     0.1000   -0.0016
    ##     40        0.8358             nan     0.1000   -0.0044
    ##     60        0.7917             nan     0.1000   -0.0037
    ##     80        0.7524             nan     0.1000   -0.0004
    ##    100        0.7195             nan     0.1000   -0.0010
    ##    120        0.6859             nan     0.1000   -0.0015
    ##    140        0.6591             nan     0.1000   -0.0026
    ##    150        0.6490             nan     0.1000   -0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0450             nan     0.1000    0.0033
    ##      2        1.0423             nan     0.1000    0.0024
    ##      3        1.0400             nan     0.1000    0.0015
    ##      4        1.0381             nan     0.1000   -0.0007
    ##      5        1.0349             nan     0.1000   -0.0015
    ##      6        1.0324             nan     0.1000    0.0020
    ##      7        1.0303             nan     0.1000   -0.0008
    ##      8        1.0270             nan     0.1000   -0.0038
    ##      9        1.0247             nan     0.1000    0.0015
    ##     10        1.0224             nan     0.1000   -0.0009
    ##     20        1.0097             nan     0.1000   -0.0009
    ##     40        0.9942             nan     0.1000   -0.0019
    ##     60        0.9829             nan     0.1000   -0.0008
    ##     80        0.9745             nan     0.1000   -0.0012
    ##    100        0.9634             nan     0.1000   -0.0020
    ##    120        0.9555             nan     0.1000   -0.0021
    ##    140        0.9480             nan     0.1000   -0.0012
    ##    150        0.9446             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0439             nan     0.1000   -0.0039
    ##      2        1.0371             nan     0.1000   -0.0017
    ##      3        1.0318             nan     0.1000   -0.0006
    ##      4        1.0217             nan     0.1000    0.0081
    ##      5        1.0177             nan     0.1000   -0.0022
    ##      6        1.0103             nan     0.1000    0.0005
    ##      7        1.0073             nan     0.1000   -0.0024
    ##      8        1.0031             nan     0.1000   -0.0006
    ##      9        0.9935             nan     0.1000   -0.0025
    ##     10        0.9904             nan     0.1000    0.0003
    ##     20        0.9597             nan     0.1000    0.0003
    ##     40        0.9220             nan     0.1000   -0.0019
    ##     60        0.8922             nan     0.1000    0.0008
    ##     80        0.8663             nan     0.1000   -0.0019
    ##    100        0.8395             nan     0.1000   -0.0016
    ##    120        0.8211             nan     0.1000   -0.0018
    ##    140        0.8029             nan     0.1000   -0.0018
    ##    150        0.7919             nan     0.1000   -0.0045
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0409             nan     0.1000    0.0004
    ##      2        1.0335             nan     0.1000   -0.0009
    ##      3        1.0257             nan     0.1000   -0.0003
    ##      4        1.0195             nan     0.1000   -0.0003
    ##      5        1.0142             nan     0.1000    0.0003
    ##      6        1.0049             nan     0.1000    0.0000
    ##      7        1.0005             nan     0.1000   -0.0030
    ##      8        0.9939             nan     0.1000    0.0008
    ##      9        0.9868             nan     0.1000   -0.0021
    ##     10        0.9752             nan     0.1000    0.0009
    ##     20        0.9345             nan     0.1000   -0.0029
    ##     40        0.8606             nan     0.1000   -0.0024
    ##     60        0.8143             nan     0.1000   -0.0008
    ##     80        0.7787             nan     0.1000   -0.0012
    ##    100        0.7409             nan     0.1000   -0.0030
    ##    120        0.7067             nan     0.1000   -0.0019
    ##    140        0.6785             nan     0.1000   -0.0018
    ##    150        0.6642             nan     0.1000   -0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0423             nan     0.1000    0.0008
    ##      2        1.0409             nan     0.1000   -0.0012
    ##      3        1.0391             nan     0.1000   -0.0019
    ##      4        1.0364             nan     0.1000   -0.0016
    ##      5        1.0320             nan     0.1000    0.0023
    ##      6        1.0303             nan     0.1000   -0.0002
    ##      7        1.0277             nan     0.1000    0.0001
    ##      8        1.0251             nan     0.1000    0.0019
    ##      9        1.0222             nan     0.1000   -0.0015
    ##     10        1.0199             nan     0.1000    0.0021
    ##     20        1.0065             nan     0.1000   -0.0022
    ##     40        0.9876             nan     0.1000   -0.0010
    ##     60        0.9782             nan     0.1000   -0.0015
    ##     80        0.9674             nan     0.1000   -0.0012
    ##    100        0.9604             nan     0.1000   -0.0017
    ##    120        0.9541             nan     0.1000   -0.0033
    ##    140        0.9465             nan     0.1000   -0.0011
    ##    150        0.9436             nan     0.1000   -0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0408             nan     0.1000    0.0026
    ##      2        1.0324             nan     0.1000   -0.0001
    ##      3        1.0284             nan     0.1000   -0.0006
    ##      4        1.0236             nan     0.1000    0.0026
    ##      5        1.0203             nan     0.1000    0.0007
    ##      6        1.0134             nan     0.1000    0.0002
    ##      7        1.0090             nan     0.1000    0.0021
    ##      8        1.0046             nan     0.1000   -0.0017
    ##      9        1.0001             nan     0.1000   -0.0019
    ##     10        0.9968             nan     0.1000   -0.0015
    ##     20        0.9650             nan     0.1000    0.0003
    ##     40        0.9221             nan     0.1000   -0.0036
    ##     60        0.8934             nan     0.1000   -0.0010
    ##     80        0.8572             nan     0.1000   -0.0015
    ##    100        0.8365             nan     0.1000   -0.0033
    ##    120        0.8094             nan     0.1000   -0.0010
    ##    140        0.7929             nan     0.1000   -0.0030
    ##    150        0.7852             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0356             nan     0.1000    0.0013
    ##      2        1.0220             nan     0.1000    0.0047
    ##      3        1.0110             nan     0.1000    0.0028
    ##      4        1.0038             nan     0.1000    0.0018
    ##      5        0.9942             nan     0.1000   -0.0027
    ##      6        0.9891             nan     0.1000   -0.0004
    ##      7        0.9831             nan     0.1000    0.0014
    ##      8        0.9798             nan     0.1000   -0.0016
    ##      9        0.9744             nan     0.1000    0.0008
    ##     10        0.9676             nan     0.1000   -0.0020
    ##     20        0.9265             nan     0.1000    0.0007
    ##     40        0.8605             nan     0.1000    0.0002
    ##     60        0.8175             nan     0.1000   -0.0031
    ##     80        0.7812             nan     0.1000   -0.0022
    ##    100        0.7482             nan     0.1000   -0.0020
    ##    120        0.7185             nan     0.1000   -0.0002
    ##    140        0.6883             nan     0.1000   -0.0026
    ##    150        0.6767             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.8501             nan     0.1000    0.0024
    ##      2        0.8462             nan     0.1000    0.0011
    ##      3        0.8438             nan     0.1000   -0.0012
    ##      4        0.8421             nan     0.1000    0.0014
    ##      5        0.8405             nan     0.1000    0.0008
    ##      6        0.8397             nan     0.1000   -0.0010
    ##      7        0.8382             nan     0.1000   -0.0003
    ##      8        0.8374             nan     0.1000   -0.0003
    ##      9        0.8356             nan     0.1000    0.0014
    ##     10        0.8336             nan     0.1000   -0.0005
    ##     20        0.8209             nan     0.1000   -0.0009
    ##     40        0.8052             nan     0.1000   -0.0002
    ##     60        0.7953             nan     0.1000   -0.0007
    ##     80        0.7887             nan     0.1000   -0.0018
    ##    100        0.7787             nan     0.1000   -0.0010
    ##    120        0.7711             nan     0.1000   -0.0008
    ##    140        0.7656             nan     0.1000   -0.0016
    ##    150        0.7636             nan     0.1000   -0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.8477             nan     0.1000    0.0030
    ##      2        0.8390             nan     0.1000    0.0031
    ##      3        0.8334             nan     0.1000    0.0025
    ##      4        0.8291             nan     0.1000    0.0006
    ##      5        0.8265             nan     0.1000   -0.0010
    ##      6        0.8172             nan     0.1000    0.0060
    ##      7        0.8140             nan     0.1000   -0.0024
    ##      8        0.8091             nan     0.1000   -0.0005
    ##      9        0.8055             nan     0.1000   -0.0003
    ##     10        0.8035             nan     0.1000    0.0002
    ##     20        0.7685             nan     0.1000   -0.0046
    ##     40        0.7324             nan     0.1000   -0.0031
    ##     60        0.7072             nan     0.1000    0.0006
    ##     80        0.6808             nan     0.1000   -0.0015
    ##    100        0.6634             nan     0.1000   -0.0008
    ##    120        0.6452             nan     0.1000   -0.0017
    ##    140        0.6327             nan     0.1000   -0.0017
    ##    150        0.6224             nan     0.1000   -0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.8418             nan     0.1000    0.0047
    ##      2        0.8358             nan     0.1000    0.0016
    ##      3        0.8206             nan     0.1000   -0.0010
    ##      4        0.8166             nan     0.1000   -0.0038
    ##      5        0.8118             nan     0.1000   -0.0012
    ##      6        0.8057             nan     0.1000    0.0005
    ##      7        0.8027             nan     0.1000    0.0003
    ##      8        0.8013             nan     0.1000   -0.0020
    ##      9        0.7950             nan     0.1000    0.0013
    ##     10        0.7881             nan     0.1000   -0.0051
    ##     20        0.7569             nan     0.1000   -0.0036
    ##     40        0.7015             nan     0.1000   -0.0018
    ##     60        0.6670             nan     0.1000   -0.0028
    ##     80        0.6287             nan     0.1000   -0.0018
    ##    100        0.5988             nan     0.1000   -0.0029
    ##    120        0.5716             nan     0.1000   -0.0002
    ##    140        0.5480             nan     0.1000   -0.0011
    ##    150        0.5389             nan     0.1000   -0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9574             nan     0.1000   -0.0003
    ##      2        0.9545             nan     0.1000    0.0021
    ##      3        0.9516             nan     0.1000    0.0022
    ##      4        0.9492             nan     0.1000    0.0004
    ##      5        0.9453             nan     0.1000   -0.0013
    ##      6        0.9438             nan     0.1000   -0.0011
    ##      7        0.9427             nan     0.1000   -0.0005
    ##      8        0.9416             nan     0.1000   -0.0010
    ##      9        0.9405             nan     0.1000   -0.0010
    ##     10        0.9391             nan     0.1000   -0.0007
    ##     20        0.9270             nan     0.1000   -0.0008
    ##     40        0.9151             nan     0.1000   -0.0011
    ##     60        0.9051             nan     0.1000    0.0003
    ##     80        0.8971             nan     0.1000   -0.0011
    ##    100        0.8895             nan     0.1000   -0.0009
    ##    120        0.8823             nan     0.1000   -0.0009
    ##    140        0.8756             nan     0.1000   -0.0009
    ##    150        0.8740             nan     0.1000   -0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9557             nan     0.1000    0.0007
    ##      2        0.9467             nan     0.1000    0.0002
    ##      3        0.9423             nan     0.1000   -0.0011
    ##      4        0.9373             nan     0.1000    0.0011
    ##      5        0.9312             nan     0.1000   -0.0001
    ##      6        0.9296             nan     0.1000    0.0001
    ##      7        0.9257             nan     0.1000   -0.0005
    ##      8        0.9222             nan     0.1000   -0.0006
    ##      9        0.9189             nan     0.1000   -0.0009
    ##     10        0.9161             nan     0.1000   -0.0017
    ##     20        0.8881             nan     0.1000   -0.0004
    ##     40        0.8424             nan     0.1000   -0.0002
    ##     60        0.8119             nan     0.1000   -0.0037
    ##     80        0.7864             nan     0.1000   -0.0016
    ##    100        0.7637             nan     0.1000   -0.0013
    ##    120        0.7464             nan     0.1000    0.0001
    ##    140        0.7242             nan     0.1000   -0.0009
    ##    150        0.7145             nan     0.1000   -0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9477             nan     0.1000    0.0020
    ##      2        0.9429             nan     0.1000    0.0005
    ##      3        0.9336             nan     0.1000   -0.0011
    ##      4        0.9292             nan     0.1000   -0.0012
    ##      5        0.9244             nan     0.1000    0.0003
    ##      6        0.9156             nan     0.1000    0.0005
    ##      7        0.9102             nan     0.1000    0.0012
    ##      8        0.9064             nan     0.1000   -0.0022
    ##      9        0.8995             nan     0.1000   -0.0011
    ##     10        0.8955             nan     0.1000    0.0018
    ##     20        0.8522             nan     0.1000   -0.0002
    ##     40        0.7821             nan     0.1000   -0.0012
    ##     60        0.7314             nan     0.1000   -0.0021
    ##     80        0.6910             nan     0.1000   -0.0005
    ##    100        0.6494             nan     0.1000   -0.0017
    ##    120        0.6212             nan     0.1000   -0.0012
    ##    140        0.6010             nan     0.1000   -0.0011
    ##    150        0.5869             nan     0.1000   -0.0004
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9852             nan     0.1000   -0.0006
    ##      2        0.9812             nan     0.1000   -0.0002
    ##      3        0.9771             nan     0.1000    0.0008
    ##      4        0.9742             nan     0.1000    0.0017
    ##      5        0.9706             nan     0.1000   -0.0010
    ##      6        0.9679             nan     0.1000    0.0012
    ##      7        0.9661             nan     0.1000    0.0004
    ##      8        0.9652             nan     0.1000   -0.0010
    ##      9        0.9627             nan     0.1000   -0.0017
    ##     10        0.9604             nan     0.1000    0.0016
    ##     20        0.9489             nan     0.1000   -0.0016
    ##     40        0.9327             nan     0.1000   -0.0005
    ##     60        0.9213             nan     0.1000   -0.0020
    ##     80        0.9138             nan     0.1000   -0.0013
    ##    100        0.9041             nan     0.1000   -0.0014
    ##    120        0.8980             nan     0.1000   -0.0019
    ##    140        0.8915             nan     0.1000   -0.0035
    ##    150        0.8869             nan     0.1000   -0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9781             nan     0.1000    0.0029
    ##      2        0.9726             nan     0.1000    0.0016
    ##      3        0.9692             nan     0.1000    0.0009
    ##      4        0.9635             nan     0.1000    0.0006
    ##      5        0.9569             nan     0.1000   -0.0014
    ##      6        0.9529             nan     0.1000   -0.0014
    ##      7        0.9505             nan     0.1000   -0.0025
    ##      8        0.9474             nan     0.1000   -0.0006
    ##      9        0.9451             nan     0.1000   -0.0011
    ##     10        0.9415             nan     0.1000   -0.0001
    ##     20        0.9114             nan     0.1000   -0.0009
    ##     40        0.8767             nan     0.1000   -0.0013
    ##     60        0.8483             nan     0.1000   -0.0034
    ##     80        0.8134             nan     0.1000   -0.0013
    ##    100        0.7853             nan     0.1000   -0.0017
    ##    120        0.7713             nan     0.1000   -0.0015
    ##    140        0.7548             nan     0.1000   -0.0026
    ##    150        0.7423             nan     0.1000   -0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9808             nan     0.1000    0.0000
    ##      2        0.9743             nan     0.1000    0.0025
    ##      3        0.9684             nan     0.1000   -0.0021
    ##      4        0.9644             nan     0.1000    0.0010
    ##      5        0.9569             nan     0.1000   -0.0013
    ##      6        0.9509             nan     0.1000    0.0014
    ##      7        0.9466             nan     0.1000    0.0005
    ##      8        0.9409             nan     0.1000   -0.0023
    ##      9        0.9356             nan     0.1000    0.0002
    ##     10        0.9294             nan     0.1000   -0.0003
    ##     20        0.8893             nan     0.1000   -0.0049
    ##     40        0.8148             nan     0.1000   -0.0015
    ##     60        0.7689             nan     0.1000   -0.0010
    ##     80        0.7254             nan     0.1000   -0.0021
    ##    100        0.6908             nan     0.1000   -0.0010
    ##    120        0.6539             nan     0.1000   -0.0005
    ##    140        0.6242             nan     0.1000   -0.0032
    ##    150        0.6141             nan     0.1000   -0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0557             nan     0.1000   -0.0005
    ##      2        1.0503             nan     0.1000   -0.0033
    ##      3        1.0477             nan     0.1000    0.0008
    ##      4        1.0452             nan     0.1000    0.0018
    ##      5        1.0437             nan     0.1000   -0.0009
    ##      6        1.0413             nan     0.1000    0.0001
    ##      7        1.0391             nan     0.1000    0.0013
    ##      8        1.0375             nan     0.1000   -0.0032
    ##      9        1.0359             nan     0.1000    0.0006
    ##     10        1.0340             nan     0.1000   -0.0011
    ##     20        1.0199             nan     0.1000   -0.0017
    ##     40        1.0044             nan     0.1000   -0.0002
    ##     60        0.9939             nan     0.1000   -0.0022
    ##     80        0.9857             nan     0.1000   -0.0008
    ##    100        0.9779             nan     0.1000   -0.0025
    ##    120        0.9697             nan     0.1000   -0.0013
    ##    140        0.9632             nan     0.1000   -0.0004
    ##    150        0.9591             nan     0.1000   -0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0542             nan     0.1000    0.0000
    ##      2        1.0429             nan     0.1000    0.0050
    ##      3        1.0372             nan     0.1000    0.0040
    ##      4        1.0326             nan     0.1000   -0.0002
    ##      5        1.0290             nan     0.1000   -0.0017
    ##      6        1.0244             nan     0.1000   -0.0023
    ##      7        1.0203             nan     0.1000   -0.0003
    ##      8        1.0168             nan     0.1000   -0.0033
    ##      9        1.0129             nan     0.1000   -0.0036
    ##     10        1.0070             nan     0.1000   -0.0003
    ##     20        0.9732             nan     0.1000   -0.0038
    ##     40        0.9295             nan     0.1000   -0.0024
    ##     60        0.8908             nan     0.1000   -0.0015
    ##     80        0.8656             nan     0.1000   -0.0021
    ##    100        0.8340             nan     0.1000   -0.0014
    ##    120        0.8093             nan     0.1000   -0.0016
    ##    140        0.7929             nan     0.1000   -0.0021
    ##    150        0.7892             nan     0.1000   -0.0031
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0423             nan     0.1000   -0.0004
    ##      2        1.0352             nan     0.1000    0.0003
    ##      3        1.0280             nan     0.1000   -0.0023
    ##      4        1.0194             nan     0.1000   -0.0033
    ##      5        1.0151             nan     0.1000    0.0002
    ##      6        1.0082             nan     0.1000   -0.0009
    ##      7        1.0031             nan     0.1000   -0.0011
    ##      8        0.9970             nan     0.1000    0.0007
    ##      9        0.9945             nan     0.1000   -0.0014
    ##     10        0.9886             nan     0.1000   -0.0035
    ##     20        0.9360             nan     0.1000   -0.0021
    ##     40        0.8635             nan     0.1000   -0.0017
    ##     60        0.8112             nan     0.1000   -0.0038
    ##     80        0.7688             nan     0.1000   -0.0029
    ##    100        0.7281             nan     0.1000   -0.0023
    ##    120        0.7013             nan     0.1000   -0.0018
    ##    140        0.6768             nan     0.1000   -0.0018
    ##    150        0.6600             nan     0.1000   -0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0625             nan     0.1000   -0.0021
    ##      2        1.0594             nan     0.1000    0.0009
    ##      3        1.0553             nan     0.1000   -0.0034
    ##      4        1.0522             nan     0.1000   -0.0004
    ##      5        1.0502             nan     0.1000    0.0008
    ##      6        1.0488             nan     0.1000   -0.0010
    ##      7        1.0461             nan     0.1000   -0.0016
    ##      8        1.0428             nan     0.1000    0.0024
    ##      9        1.0407             nan     0.1000   -0.0017
    ##     10        1.0379             nan     0.1000    0.0019
    ##     20        1.0244             nan     0.1000   -0.0021
    ##     40        1.0091             nan     0.1000   -0.0007
    ##     60        0.9951             nan     0.1000   -0.0034
    ##     80        0.9834             nan     0.1000   -0.0040
    ##    100        0.9763             nan     0.1000    0.0000
    ##    120        0.9680             nan     0.1000   -0.0021
    ##    140        0.9621             nan     0.1000   -0.0024
    ##    150        0.9604             nan     0.1000   -0.0042
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0602             nan     0.1000   -0.0002
    ##      2        1.0544             nan     0.1000   -0.0010
    ##      3        1.0458             nan     0.1000   -0.0018
    ##      4        1.0420             nan     0.1000   -0.0015
    ##      5        1.0354             nan     0.1000    0.0027
    ##      6        1.0324             nan     0.1000    0.0013
    ##      7        1.0267             nan     0.1000   -0.0004
    ##      8        1.0207             nan     0.1000   -0.0015
    ##      9        1.0148             nan     0.1000   -0.0015
    ##     10        1.0103             nan     0.1000   -0.0032
    ##     20        0.9756             nan     0.1000    0.0010
    ##     40        0.9331             nan     0.1000   -0.0012
    ##     60        0.8988             nan     0.1000   -0.0015
    ##     80        0.8657             nan     0.1000   -0.0010
    ##    100        0.8446             nan     0.1000   -0.0037
    ##    120        0.8247             nan     0.1000   -0.0066
    ##    140        0.8020             nan     0.1000   -0.0024
    ##    150        0.7977             nan     0.1000   -0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.0614             nan     0.1000    0.0045
    ##      2        1.0533             nan     0.1000    0.0042
    ##      3        1.0451             nan     0.1000   -0.0019
    ##      4        1.0390             nan     0.1000   -0.0020
    ##      5        1.0357             nan     0.1000   -0.0020
    ##      6        1.0262             nan     0.1000    0.0016
    ##      7        1.0216             nan     0.1000   -0.0012
    ##      8        1.0177             nan     0.1000   -0.0016
    ##      9        1.0138             nan     0.1000   -0.0009
    ##     10        1.0054             nan     0.1000    0.0011
    ##     20        0.9444             nan     0.1000   -0.0044
    ##     40        0.8750             nan     0.1000    0.0010
    ##     60        0.8325             nan     0.1000   -0.0025
    ##     80        0.7843             nan     0.1000   -0.0018
    ##    100        0.7478             nan     0.1000   -0.0017
    ##    120        0.7173             nan     0.1000   -0.0022
    ##    140        0.6764             nan     0.1000   -0.0019
    ##    150        0.6627             nan     0.1000   -0.0028
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        0.9961             nan     0.1000    0.0018
    ##      2        0.9934             nan     0.1000    0.0018
    ##      3        0.9907             nan     0.1000   -0.0006
    ##      4        0.9888             nan     0.1000   -0.0015
    ##      5        0.9860             nan     0.1000    0.0005
    ##      6        0.9838             nan     0.1000   -0.0003
    ##      7        0.9814             nan     0.1000   -0.0001
    ##      8        0.9800             nan     0.1000    0.0003
    ##      9        0.9787             nan     0.1000    0.0004
    ##     10        0.9767             nan     0.1000   -0.0013
    ##     20        0.9626             nan     0.1000   -0.0002
    ##     40        0.9479             nan     0.1000   -0.0018
    ##     50        0.9411             nan     0.1000   -0.0009

    gbmFit

    ## Stochastic Gradient Boosting 
    ## 
    ## 1917 samples
    ##   37 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1725, 1724, 1725, 1726, 1726, 1725, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE       Rsquared    MAE      
    ##   1                   50      0.9655051  0.02141015  0.5040835
    ##   1                  100      0.9669151  0.01936185  0.5055205
    ##   1                  150      0.9721854  0.02039711  0.5075027
    ##   2                   50      0.9704233  0.01641829  0.5105550
    ##   2                  100      0.9821552  0.01387870  0.5148266
    ##   2                  150      0.9871269  0.01573674  0.5189850
    ##   3                   50      0.9838136  0.01057260  0.5124071
    ##   3                  100      0.9962549  0.01073196  0.5226253
    ##   3                  150      1.0075454  0.01262158  0.5336594
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value
    ##  of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

    # create the prediction
    pred2 <- predict(gbmFit, newdata = newsPopTest)

    # compare the prediction vs the actual
    resample2 <- postResample(pred2, obs = newsPopTest$shares)
    resample2

    ##       RMSE   Rsquared        MAE 
    ## 1.13415711 0.01669614 0.51996982

### Linear Regression Model

Linear regression is used to predict the outcome of a response variable
for 1 to n predictors. The aim is to establish a linear relationship
between the predictor variable(s) and response variable so we can
predict the value of the response when only the predictor variable(s)
is(are) known.

    # train the linear model for main effects + interactions on first 3 preds
    lmFit <- train(shares ~ timedelta*n_tokens_title*n_tokens_content, data = newsPopTrain,
                                                                       method = "lm", preProces = c("center", "scale"),
                                                                       trControl = trainControl(method = "cv", number = 10))
    lmFit

    ## Linear Regression 
    ## 
    ## 1917 samples
    ##    3 predictor
    ## 
    ## Pre-processing: centered (7), scaled (7) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1726, 1724, 1726, 1724, 1726, 1725, ... 
    ## Resampling results:
    ## 
    ##   RMSE       Rsquared  MAE      
    ##   0.9764353  0.013713  0.5116786
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

    # create the prediction
    pred3 <- predict(lmFit, newdata = newsPopTest)

    # compare the prediction vs the actual
    resample3 <- postResample(pred3, obs = newsPopTest$shares)
    resample3

    ##         RMSE     Rsquared          MAE 
    ## 1.1523325022 0.0003277941 0.5283805845

### Comparison

Below is a comparison of the 3 methods. All have relatively high root
mean square errors.

    # compare results from 3 methods
    comparison <- data.frame("RSME" = c(resample1[[1]], resample2[[1]], resample3[1]), "MAE" = c(resample1[[3]], resample2[[3]], resample3[[3]]))
    rownames(comparison) <- c("RPART","GBM", "LM")
    kable(comparison)

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
1.143403
</td>
<td style="text-align:right;">
0.5214574
</td>
</tr>
<tr>
<td style="text-align:left;">
GBM
</td>
<td style="text-align:right;">
1.134157
</td>
<td style="text-align:right;">
0.5199698
</td>
</tr>
<tr>
<td style="text-align:left;">
LM
</td>
<td style="text-align:right;">
1.152332
</td>
<td style="text-align:right;">
0.5283806
</td>
</tr>
</tbody>
</table>
