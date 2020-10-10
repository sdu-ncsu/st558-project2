## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/sdu-ncsu/st558-project2/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

## Thoughts

## What Would I Do Differently?

## What Was the Most Difficult Part for me?

## What are My Big Take-aways from this Project?

## Github Repo and Github Pages URL

https://github.com/sdu-ncsu/st558-project2

https://sdu-ncsu.github.io/st558-project2/

## Introduction

This will be an exercise in analyzing online news using machine learning techniques from ST558. The purpose of the analysis is to create a model which predicts the popularity of the article based on certain attributes of the article. For this purpose, the number of shares will be used as a gauge of popularity. The two methods which will be used are a non-ensembeled decision tree and an ensembeled boosted tree model. In both cases cross validation will be used and the final model used on a test data set which will be 30% of the data of the initial data set. The variables which will be used in the model are:


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


### Test link

Analysis for this data will be done for different days of the week.

The analysis for [Monday is available here](linkOne.md).
