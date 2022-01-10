from flask import Flask, request, jsonify
import pickle
import numpy as np
from collections import defaultdict,Counter
import math
import re
import heapq
from inverted_index_gcp import *
import time
import json


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        with open('chamber_of_secrets/postings_gcp/all_stopwords.pkl', 'rb') as f:
            self.all_stopwords = pickle.load(f)

        # reading title_index
        with open('chamber_of_secrets/postings_gcp/title_index.pkl', 'rb') as f:
            self.title_index = pickle.load(f)

        # reading text_index
        with open('chamber_of_secrets/postings_gcp/text_index.pkl', 'rb') as f:
            self.text_index = pickle.load(f)

        # reading anchor_index
        with open('chamber_of_secrets/postings_gcp/anchor_index.pkl', 'rb') as f:
            self.anchor_index = pickle.load(f)

        # reading page_views
        with open('chamber_of_secrets/postings_gcp/pageviews-202108-user.pkl', 'rb') as f:
            self.wid2pv = pickle.load(f)
            self.pv_max, self.pv_min, self.pv_mean = (max(self.wid2pv), min(self.wid2pv), np.mean(list(self.wid2pv.values())))
        # reading pagerank
        with open('chamber_of_secrets/postings_gcp/id_pagerank.pkl', 'rb') as f:
            self.pagerank = pickle.load(f)
            self.pr_max, self.pr_min, self.pr_mean = (max(self.pagerank), min(self.pagerank), np.mean(list(self.pagerank.values())))

        # reading titles
        with open('chamber_of_secrets/postings_gcp/id_title.pkl', 'rb') as f:
            self.id_title = pickle.load(f)

        #read train queries, for evaluaion purposes only
        with open("queries_train.json") as jsonFile:
            self.queries_with_answers = json.load(jsonFile)
            jsonFile.close()

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    # tokenize the query with the same tokenizer that build the index
    query = tokenize_query(query)
    if len(query) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    # cosine similarity over body
    cosine_similarity_dict = cosine_similarity(query, app.text_index)
    top_100 = get_top_n(cosine_similarity_dict, 100)
    cosine_result = [int(id) for id, score in top_100]
    # groups to reorder by pagerank and pageview
    first_40 = cosine_result[:40]
    last_60 = cosine_result[40:]
    # reorder results according to page rank and page view (with weights)
    reorder_result_40 = consider_pr_pv(first_40, pagerank_w=0.6, pageview_w=0.4, pagerank_dict=app.pagerank,
                                         pageview_dict=app.wid2pv)
    # add titles and combine
    res = [(int(id), app.id_title[id]) for id, score in reorder_result_40]
    res += [(int(id), app.id_title[id]) for id in last_60]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    # change this before submission
    query = request.args.get('query', '')
    # tokenize the query with the same tokenizer that build the index
    query = tokenize_query(query)
    if len(query) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    # calculate cosine similarity
    cosine_similarity_dict = cosine_similarity(query, app.text_index)
    # get top 100
    top_100 = get_top_n(cosine_similarity_dict, 100)
    # convert format to (wiki_id, title)
    res =[(int(id), app.id_title[id]) for id, score in top_100]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    # tokenize the query with the same tokenizer that build the index
    query = tokenize_query(query)
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    id_tf = []
    for token in query:
        if token in app.title_index.df.keys():
            id_tf += app.title_index.read_posting_list(token)  # get posting lists
    ids = list(map(lambda x: (x[0]), id_tf))  # map to doc_id
    id_qf = list(Counter(ids).items())  # [(doc_id,sum of words in the query),....)]
    pl = sorted(id_qf, key=lambda x: x[1], reverse=True)  # sorted by value
    ids = list(map(lambda x: x[0],pl))  # map top doc_id
    res =[(id, app.id_title[id]) for id in ids]  # add titles
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    query = tokenize_query(query)
    id_tf = []
    for token in query:
        if token in app.anchor_index.df.keys():
            id_tf += app.anchor_index.read_posting_list(token)  # get posting lists
    ids = list(map(lambda x: (x[0]), id_tf))  # map to doc_id
    id_qf = list(Counter(ids).items())  # [(doc_id,sum of words in the query),....)]
    pl = sorted(id_qf, key=lambda x: x[1], reverse=True)  # sorted by value
    ids = list(map(lambda x: x[0],pl))  # map to doc_id
    res =[(id, app.id_title[id]) for id in ids]  # add titles
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # retrieve pagerank for each given id
    res = [app.pagerank[id] for id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    # change this before submission
    if len(wiki_ids) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    # retrieve pageview for each given id
    for id in wiki_ids:
        res.append(app.wid2pv[id])
    # END SOLUTION
    return jsonify(res)

#HELPER - combine results from SAME calculation method - NOT USED
def merge_results(scores_1, scores_2, scores_1_weight=0.5, scores_2_weight=0.5, N=40):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: list of pairs in the following format:(doc_id,score)

    body_scores: list of pairs in the following format:(doc_id,score)

    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores

    N: Integer. How many document to retrieve. This argument is passed to topN function. for the topN function.

    Returns:
    -----------
    list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE
    merged_result = defaultdict(lambda: 0)
    for doc_id, score_1 in scores_1:
        merged_result[doc_id] += score_1 * scores_1_weight
    for doc_id, score_2 in scores_2:
        merged_result[doc_id] += score_2 * scores_2_weight

    sorted_result = heapq_nlargest(merged_result.items(), N)
    return sorted_result

#HELPER - get k largest sorted, sort at o(nlog(k) + klog(k)) instead of o(nlog(n))
def heapq_nlargest(unsorted, k):
    return sorted(heapq.nlargest(k, unsorted, key= lambda x: x[1]), key=lambda x:x[1], reverse=True)

#HELPER - retrieve top N from similarity dictionary
def get_top_n(sim_dict, N):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    unsorted = [(doc_id, np.round(score, 5)) for doc_id, score in sim_dict.items()]
    return heapq_nlargest(unsorted, N)

#HELPER - cosine similarity calculation
def cosine_similarity(query, index):
    cos_dict = {}
    # generate query tfidf - [(term, tfidf), (term, tfidf)...]
    query_tfidf = generate_query_tfidf(query, index)
    # query normalize factor for cosine
    nf_q = math.sqrt(sum([tf_idf ** 2 for term, tf_idf in query_tfidf]))
    # calculate doc tfidf and multiply with query tfidf (numerator)
    for term, tf_idf in query_tfidf:
        if term in index.df.keys():
            pls = index.read_posting_list(term)
            for doc_id, freq in pls:
                try:
                    tf = freq / index.dl[doc_id]
                    idf = np.log10(len(index.dl) / index.df[term])
                    doc_tfidf = tf * idf * tf_idf
                    cos_dict[doc_id] = cos_dict.get(doc_id, 0) + doc_tfidf
                except:
                    cos_dict[doc_id] = 0
    # calculate denominator and combine with numerator
    for doc_id in cos_dict.keys():
        try:
            cos_dict[doc_id] = cos_dict[doc_id] * (1/nf_q) * (1/(index.nf_docs[doc_id]))
        except:
            cos_dict[doc_id] = 0
    return cos_dict

#HELPER - cosine similarity
def generate_query_tfidf(query_to_search, index):
    epsilon = .0000001
    Q = []
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.dl)) / (df + epsilon), 10)  # smoothing
            try:
                Q.append((token, tf*idf))
            except:
                pass
    return Q

#HELPER - clean and tokenize a query
def tokenize_query(query):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [item for item in tokens if item not in app.all_stopwords]
    return tokens

#HELPER - sort doc_ids by weighting pagerank and pageview
def consider_pr_pv(docs, pagerank_w , pageview_w , pagerank_dict , pageview_dict):
    sorted_by_weigts = {}
    for doc_id in docs:
        # normalize by minmax
        pageview = (pageview_dict.get(doc_id, 680) - app.pv_mean) / (app.pv_max - app.pv_min)
        pagerank = (pagerank_dict.get(doc_id, 1) - app.pr_mean) / (app.pr_max - app.pr_min)
        sorted_by_weigts[doc_id] = pagerank_w * pagerank + pageview_w * pageview
        #no need to use heapq, 40 values to sort
    return sorted(list(sorted_by_weigts.items()), key=lambda x: x[1], reverse=True)

#HELPER Evaluation - internal use
@app.route("/report")
def report():
    query = request.args.get('query', '')
    quaries_with_answers = dict(app.queries_with_answers)
    queries = list(quaries_with_answers.keys())
    scores = []
    methods = [search_offline]
    for method in methods:
        t0 = time.time()
        for query in queries:
            rel_docs_scores = method(query)
            predicted_docs = [doc_id for doc_id, title in rel_docs_scores]
            true_docs = quaries_with_answers[query]
            map40 = map_at_k(true_docs, predicted_docs)
            scores.append((query, map40))
        mean_score = np.mean([score for query, score in scores])
        t1 = time.time()
        total = t1-t0
        scores.append(('mean score', mean_score, total))
    return jsonify(scores)

#HELPER - Evaluation - internal use
def search_offline(query):
    # tokenize the query with the same tokenizer that build the index
    query = tokenize_query(query)
    # cosine similarity over body
    cosine_similarity_dict = cosine_similarity(query, app.text_index)
    top_100 = get_top_n(cosine_similarity_dict, 100)
    cosine_result = [int(id) for id, score in top_100]
    #groups to reorder by pagerank and pageview
    first_40 = cosine_result[:40]
    last_60 = cosine_result[40:]
    # reorder results according to page rank and page view (with weights)
    reorder_result_40 = consider_pr_pv(first_40, pagerank_w=0.6, pageview_w=0.4, pagerank_dict=app.pagerank,
                                       pageview_dict=app.wid2pv)
    # add titles and combine
    res = [(int(id), app.id_title[id]) for id, score in reorder_result_40]
    res += [(int(id), app.id_title[id]) for id in last_60]
    return res

#HELPER - report - internal use
def intersection(l1,l2):
    return list(set(l1)&set(l2))

#HELPER - report - internal use
def precision_at_k(true_list, predicted_list, k=40):
    return round(len(intersection(true_list, predicted_list[:k])) / k, 3)

#HELPER - report - internal use
def map_at_k(true_list, predicted_list, k=40):
    relevant_indexes_k = [i + 1 for i in range(len(predicted_list[:k])) if predicted_list[i] in true_list]
    p_k_rel_k = [precision_at_k(true_list, predicted_list, i) for i in relevant_indexes_k]
    try:
        return round(sum(p_k_rel_k) / len(p_k_rel_k), 3)
    except:
        return round(0, 3)



if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

