# IR-engine
Information Retrieval course project.

By Omer Rosenberg and Chen Harel




## Inverted Index GCP
The core of this file has been taken from assignment3, we added some features in it, in order to minimize the query response time.

nf_docs - normalize factor for docs tf-idf.

dl - a dictionary of document length foreach doc.

bm25_idf - calculating BM25-IDF.

## Index Maker GCP
Implamentation of the functionality of the features we add to the Inverted-Index class, we used Map-reduce functions with pyspark library in order to make it faster and parralelize it in Google cloud platform.

## All Stop-words
A pikle document which contains all the stop word we would like to eliminate in our tokenisation step for the query.


## Search Frontend
get_pageview - Returns the number of page views that each of the provide wiki articles had in August 2021.

get_pagerank - Returns PageRank values for a list of provided wiki article IDs.

search_anchor - Returns all search result that contain a query word in the anchor text of the title

search_title - Returns all search result that contain a query word in the title of articles.

search_body - Returns up to 100 search results for the query using TF-IDF and cosine-similarity of the "BODY" of articles.

search - Returns our best searching result for a query, using a fusion of cosine-similarity, Pagerank and Pageviews of an articles, and weighted them with weighting function, after finding the optimal weights in a greed search.







