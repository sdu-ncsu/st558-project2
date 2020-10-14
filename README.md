News Popularity Monday Data
================
Shuang Du
10/16/2020

# ST558–Project 2 News Popularity

## Libraries Used

``` r
# library(readxl);
# library(tidyverse);
# library(caret);
# library(modelr);
# library(rpart);
# library(kableExtra);
```

## Create All Pages

``` r
# rmarkdown::render("monday.Rmd", params = list(day = 'monday'))
# rmarkdown::render("tuesday.Rmd", params = list(day = 'tuesday'))
# rmarkdown::render("wednesday.Rmd", params = list(day = 'wednesday'))
# rmarkdown::render("thursday.Rmd", params = list(day = 'thursday'))
# rmarkdown::render("friday.Rmd", params = list(day = 'friday'))
# rmarkdown::render("saturday.Rmd", params = list(day = 'saturday'))
# rmarkdown::render("sunday.Rmd", params = list(day = 'sunday'))
```

## Links to Various Pages of Analysis

Analysis for this data will be done for different days of the week.

  - The analysis for [Monday is available here](monday.md).
  - The analysis for [Tuesday is available here](tuesday.md).
  - The analysis for [Wednesday is available here](wednesday.md).
  - The analysis for [Thursday is available here](thursday.md).
  - The analysis for [Friday is available here](friday.md).
  - The analysis for [Saturday is available here](saturday.md).
  - The analysis for [Sunday is available here](sunday.md).

## Thoughts

Given more time, I may have tried to pass the date filtered data frame
as a parameter to the seven files. In this manner, the function call
would have only been needed once. The most difficult part of this
project is trying to understand what a good result is in terms of
prediction is an RSME of about .8 good or bad? Additionally, it occurred
to me that without field specific knowledge of, in this case news
popularity, its almost impossible to guess at what parameters may or may
not have any impact on the final result. Consequently, one is left with
just randomly picking variables that seem reasonable and attempting to
model. My major takeaway from this project is that some research into
the topic at hand is crucial for both creating the model and judging the
model’s merits.

## Github Repo and Github Pages URL

<https://github.com/sdu-ncsu/st558-project2>

<https://sdu-ncsu.github.io/st558-project2/>

## Introduction

This will be an exercise in analyzing online news using machine learning
techniques from ST558. The purpose of the analysis is to create a model
which predicts the popularity of the article based on certain attributes
of the article. For this purpose, the number of shares will be used as a
gauge of popularity. The two methods which will be used are a
non-ensembeled decision tree and an ensembeled boosted tree model. In
both cases cross validation will be used and the final model used on a
test data set which will be 30% of the data of the initial data set. The
variables which will be used in the model are:

``` 
 1. timedelta:                     Days between the article publication and
                                   the dataset acquisition
 2. n_tokens_title:                Number of words in the title
 3. n_tokens_content:              Number of words in the content
 4. n_unique_tokens:               Rate of unique words in the content
 5. n_non_stop_words:              Rate of non-stop words in the content
 6. n_non_stop_unique_tokens:      Rate of unique non-stop words in the
                                   content
 7. num_hrefs:                     Number of links
 8. num_self_hrefs:                Number of links to other articles
                                   published by Mashable
 9. num_imgs:                      Number of images
10. num_videos:                    Number of videos
11. average_token_length:          Average length of the words in the
                                   content
12. num_keywords:                  Number of keywords in the metadata
13. data_channel_is_lifestyle:     Is data channel 'Lifestyle'?
14. data_channel_is_entertainment: Is data channel 'Entertainment'?
15. data_channel_is_bus:           Is data channel 'Business'?
16. data_channel_is_socmed:        Is data channel 'Social Media'?
17. data_channel_is_tech:          Is data channel 'Tech'?
18. data_channel_is_world:         Is data channel 'World'?
19. self_reference_min_shares:     Min. shares of referenced articles in
                                   Mashable
20. self_reference_max_shares:     Max. shares of referenced articles in
                                   Mashable
21. self_reference_avg_sharess:    Avg. shares of referenced articles in
                                   Mashable
22. global_subjectivity:           Text subjectivity
23. global_sentiment_polarity:     Text sentiment polarity
24. global_rate_positive_words:    Rate of positive words in the content
25. global_rate_negative_words:    Rate of negative words in the content
26. rate_positive_words:           Rate of positive words among non-neutral
                                   tokens
27. rate_negative_words:           Rate of negative words among non-neutral
                                   tokens
28. avg_positive_polarity:         Avg. polarity of positive words
29. min_positive_polarity:         Min. polarity of positive words
30. max_positive_polarity:         Max. polarity of positive words
31. avg_negative_polarity:         Avg. polarity of negative  words
32. min_negative_polarity:         Min. polarity of negative  words
33. max_negative_polarity:         Max. polarity of negative  words
34. title_subjectivity:            Title subjectivity
35. title_sentiment_polarity:      Title polarity
36. abs_title_subjectivity:        Absolute subjectivity level
37. abs_title_sentiment_polarity:  Absolute polarity level
38. shares:                        Number of shares (target)
```
