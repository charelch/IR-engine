# IR-engine
Information Retrieval course project.

By Omer Rosenberg and Chen Harel




## Inverted Index GCP
The core of this file has been taken from assignment3, we added some features in it, in order to minimize the query response time.

nf_docs - normalize factor for docs tf-idf.

dl - a dictionary of doc length foreach doc.

bm25_idf - calculating BM25-IDF.

## Index Maker GCP
Implamentation of the functionality of the features we add to the Inverted-Index class, we used Map-reduce functions with pyspark library in order to make it faster and parralelize it in Google cloud platform.

## All Stop-words
A pikle document which contains all the stop word we would like to eliminate in our tokenisation step for the query.


## Search Frontend



