# Cryptocurrency Recommendations
This project develops a cryptocurrency recommendation engine (https://cryptocurrency-recommendations.onrender.com), which recommends cryptocurrency projects for users based on what they want to invest in.

## Data
Data for the recommendation engine was acquired online from the following sources:
+ [Cryptocompare](https://www.cryptocompare.com/coins/list/all/USD/1)
+ [CoinMarketCap](https://coinmarketcap.com/)
+ [Kraken](https://support.kraken.com/hc/en-us/articles/203325283-Cryptocurrency-deposit-processing-times)
+ [CER](https://cer.live/cryptocurrency-security-rating)
+ [Cryptowisser](https://www.cryptowisser.com/crypto-carbon-footprint)

In addition, recommendations provided a link to project whitepapers (if readily available), which were sourced from:
+ [All Crypto White Papers](https://www.allcryptowhitepapers.com/)

## Pipeline
The data processing and analysis steps of the pipeline are described more in-depth on the ["How it works"](https://cryptocurrency-recommendations.onrender.com/howitworks.html) section of the website, but here is an outline of the process and indicates the files used in each step.
+ <b>Data Acquisition:</b> Relevant information from these sources was gathered via APIs and scraping using files in the <code>acquisition.py</code> file, and is stored in intermediate files in the coin_data folder.<br><br>
+ <b>Data Processing and Analysis:</b> Data from each source was processed by functions in the <code>feature_matrix.py</code> file. When this involves non-trivial processing of text (i.e., project descriptions), functions in <code>text_analysis.py</code> are also used.<br><br>
+ <b>Feature Matrix Construction:</b> The feature matrix was constructed from processed data using the <code>build_feature_matrix()</code> function and stored as a JSON file. Relevant information included with each recommendation (i.e., project links and whitepapers) is also stored in a JSON created by the <code>create_recommendation_json()</code> function. Both of these functions are also in the <code>feature_matrix.py</code> file.<br><br>
+ <b>Finding Recommendations:</b> The recommendation engine may be tested using the <code>find_recs()</code> function in the <code>recommendations.py</code> file. For each recommendation, the relevant information to provide to the user may also be accessed with the <code>read_recommendation_json()</code> function.<br><br>
+ <b>Visualization:</b> Data visualization helped inform the other steps of the pipeline, and some examples are included on the website. The code for this is included in the <code>visualizations.py</code> file.
