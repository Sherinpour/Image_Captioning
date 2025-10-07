## Data Overview

- **Rows**: 378341

- **Columns**: 13

- **Memory usage (MB)**: 450.21



### Missing values by column

|                |   missing |
|:---------------|----------:|
| gender         |    334343 |
| persian_name   |    109467 |
| brand          |     66012 |
| group          |         5 |
| title          |         0 |
| image_url      |         0 |
| entities       |         0 |
| product        |         0 |
| _source_file   |         0 |
| random_key     |         0 |
| title_length   |         0 |
| product_length |         0 |
| group_length   |         0 |



## Descriptive Statistics (Numerical)

|                |   count |     mean |      std |   min |   5% |   25% |   50% |   75% |   95% |   max |
|:---------------|--------:|---------:|---------:|------:|-----:|------:|------:|------:|------:|------:|
| title_length   |  378341 | 42.1597  | 22.1499  |     2 |   14 |    26 |    39 |    53 |    84 |   950 |
| product_length |  378341 | 10.0862  |  5.63477 |     1 |    3 |     6 |     9 |    13 |    20 |   117 |
| group_length   |  378341 |  6.12431 |  1.89259 |     1 |    5 |     6 |     6 |     6 |     7 |    67 |



## Key Frequencies

### Top groups

| value            |   count |
|:-----------------|--------:|
| نامشخص           |  173926 |
| زنانه            |   86114 |
| مردانه           |   81524 |
| دخترانه          |   11551 |
| پسرانه           |    4121 |
| بچگانه           |    4104 |
| زنانه و دخترانه  |    2067 |
| مردانه و زنانه   |    1979 |
| مردانه زنانه     |     942 |
| بچه گانه         |     735 |
| زنانه و مردانه   |     661 |
| زنانه دخترانه    |     648 |
| دخترانه و زنانه  |     600 |
| خردسال و نوجوان  |     438 |
| کودک             |     375 |
| پسرانه – دخترانه |     313 |
| بارداری          |     282 |
| دخترانه و پسرانه |     275 |
| یونیسکس          |     218 |
| کودک / نوجوان    |     209 |


### Top products

| value           |   count |
|:----------------|--------:|
| پیراهن          |   20711 |
| کفش             |   10022 |
| تیشرت           |    7666 |
| ساعت مچی        |    6272 |
| شلوار           |    5699 |
| جاکلیدی         |    5263 |
| مانتو           |    5065 |
| کاپشن           |    4747 |
| سویشرت          |    4414 |
| ساعت            |    3715 |
| شلوار جین       |    3337 |
| کیف پول         |    3136 |
| آینه            |    3101 |
| آینه جیبی       |    3021 |
| هودی            |    2899 |
| کیف لوازم آرایش |    2877 |
| آینه تاشو       |    2770 |
| شورت            |    2635 |
| توت بگ پارچه ای |    2376 |
| تی شرت          |    2316 |



## Relationships and Correlations

![Correlation Heatmap](/home/sherin/Image_Captioning/eda_project/scripts/figures/correlation_heatmap.png)


## Missing Data Visualizations

![Missingness](/home/sherin/Image_Captioning/eda_project/scripts/figures/missing_matrix.png)

![Missingness](/home/sherin/Image_Captioning/eda_project/scripts/figures/missing_bar.png)


## Persian Text Analysis

### Top tokens in title

| token    |   count |
|:---------|--------:|
| مردانه   |   88657 |
| زنانه    |   76126 |
| کد       |   58397 |
| مدل      |   53060 |
| طرح      |   46709 |
| شلوار    |   35351 |
| اورجینال |   35219 |
| پیراهن   |   28651 |
| ای       |   28526 |
| برند     |   26273 |
| کیف      |   25808 |
| ست       |   24781 |
| کفش      |   23831 |
| سایز     |   22622 |
| خندالو   |   21828 |
| رنگ      |   20850 |
| ساعت     |   20459 |
| مشکی     |   20416 |
| تیشرت    |   19369 |
| دار      |   18834 |
| آستین    |   18459 |
| دخترانه  |   16642 |
| بلند     |   12496 |
| مانتو    |   12103 |
| شورت     |   11741 |
| لباس     |   11198 |
| آبی      |   11022 |
| بند      |   10722 |
| چرم      |   10673 |
| آینه     |   10433 |
| مجلسی    |   10325 |
| مناسب    |   10300 |
| پارچه    |   10070 |
| سفید     |   10054 |
| شومیز    |    9667 |
| تی       |    9488 |
| کاپشن    |    9442 |
| کوتاه    |    9328 |
| یقه      |    9297 |
| جین      |    8765 |
| شرت      |    8364 |
| نیم      |    8215 |
| سنگ      |    8200 |
| سویشرت   |    8161 |
| وایکیکی  |    8112 |
| جاکلیدی  |    8086 |
| کت       |    7987 |
| بلوز     |    7722 |
| نگین     |    7625 |
| پول      |    7370 |


![Title Wordcloud](/home/sherin/Image_Captioning/eda_project/scripts/figures/title_wordcloud.png)


## Notes

- Frequency tables are top-20 by count unless otherwise noted.

- Text normalized for Persian orthography; minimal stopword removal applied.