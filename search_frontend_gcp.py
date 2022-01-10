from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict,Counter
import math
import re
import builtins as bu
import heapq
from inverted_index_gcp import *
#report - evaluation
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

        # reading pagerank
        with open('chamber_of_secrets/postings_gcp/id_pagerank.pkl', 'rb') as f:
             self.pagerank = pickle.load(f)

        # reading titles
        with open('chamber_of_secrets/postings_gcp/id_title.pkl', 'rb') as f:
             self.id_title = pickle.load(f)

        with open("queries_train.json") as jsonFile:
            self.queries_with_answers = json.load(jsonFile)
            jsonFile.close()

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

#search final
@app.route("/search")
# change this before submission
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
    query = tokenize_query(query)
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

#search - version 1, binary dot product
def search_version_1(query):
    text_index = app.text_index
    query = tokenize_query(query)
    words, pls = get_posting_gen(text_index)
    binary_df = generate_document_binary_df(query, words, pls)
    binary_matrix = binary_df.to_numpy()
    q_vector = generate_query_binary_vector(query, text_index)
    dot_product_score_vector = q_vector.dot(binary_matrix.transpose())
    indices = (dot_product_score_vector).argsort()[:12]
    best = []
    for i in indices:
        best.append(int(binary_df.transpose().columns[i]))
    res =[(id, app.id_title[id]) for id in best]
    return res

@app.route("/search_body")
# change this before submission
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
    query = tokenize_query(query)

    if len(query) == 0:
      return jsonify(res)

    # BEGIN SOLUTION
    t2 = time.time()
    cosine_similarity_dict = cosine_similarity(query, app.text_index)
    #cosine_similarity_dict = CosineSimilarity_body(app.text_index, query)
    t3 = time.time()
    delta_cosine = t3-t2
    top_100 = get_top_n(cosine_similarity_dict, 100)
    res =[(int(id), app.id_title[id]) for id, score in top_100]

    res.append(('cosine', delta_cosine))
    # END SOLUTION
    return jsonify(res)
#SPARK
@app.route("/search_title")
# change this before submission
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
    ####change this before submission
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query = tokenize_query(query)
    id_tf = []
    for token in query:
        if token in app.title_index.df.keys():
            id_tf += app.title_index.read_posting_list(token)
    ids = list(map(lambda x: (x[0]), id_tf))
    id_qf = list(Counter(ids).items())  # [(doc_id,sum of words in the query),....)]
    pl = sorted(id_qf, key=lambda x: x[1], reverse=True)  # sorted by value
    ids = list(map(lambda x: x[0],pl))
    res =[(id, app.id_title[id]) for id in ids]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
# change this before submission
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
            id_tf += app.anchor_index.read_posting_list(token)
    ids = list(map(lambda x: (x[0]), id_tf))
    id_qf = list(Counter(ids).items())  # [(doc_id,sum of words in the query),....)]
    pl = sorted(id_qf, key=lambda x: x[1], reverse=True)  # sorted by value
    ids = list(map(lambda x: x[0],pl))
    res =[(id, app.id_title[id]) for id in ids]
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
    #wiki_ids = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    #change this before submission - spark and read pagerank
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

    for id in wiki_ids:
        res.append(app.wid2pv[id])
    # END SOLUTION
    return jsonify(res)

#HELPER - weight 2 indexes
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

#HELPER - weight 3 indexes
def merge_resultsV2(scores_1, scores_2,scores_3, scores_1_weight=1/3, scores_2_weight=1/3,scores_3_weight=1/3, N=40):
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
    for doc_id, score in scores_1:
        merged_result[doc_id] += score * scores_1_weight
    for doc_id, score in scores_2:
        merged_result[doc_id] += score * scores_2_weight
    for doc_id, score in scores_3:
        merged_result[doc_id] += score * scores_3_weight

    sorted_result = heapq_nlargest(merged_result.items(), N)
    return sorted_result

#HELPER - get k largest sorted
def heapq_nlargest(unsorted, k):
    return sorted(heapq.nlargest(k, unsorted, key= lambda x: x[1]), key=lambda x:x[1], reverse=True)

#search body, search version 1 HELPER
def get_posting_gen(index):
    """
    This function returning the generator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls

#search_body HELPER
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

def cosine_similarity(query, index):
    cos_dict = {}
    query_tfidf = generate_query_tfidf_vector(query, index)
    nf_q = math.sqrt(sum([tf_idf ** 2 for term, tf_idf in query_tfidf]))
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
    for doc_id in cos_dict.keys():
        try:
            cos_dict[doc_id] = cos_dict[doc_id] * (1/nf_q) * (1/(index.nf_docs[doc_id]))
        except:
            cos_dict[doc_id] = 0
    return cos_dict

#search body HELPER
def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    #Q = np.zeros(len(query_to_search))
    Q = []
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.dl)) / (df + epsilon), 10)  # smoothing
            try:
                Q.append((token, tf*idf))
                #ind = query_to_search.index(token)
                #Q[ind] = tf * idf
            except:
                pass
    return Q

#search version 1 HELPER
def generate_query_binary_vector(query_to_search, index):
    Q = np.zeros(len(query_to_search))
    for token in query_to_search:
        if token in index.df.keys():
            Q[query_to_search.index(token)] = 1
    return Q

#search_version 1 HELPER
def get_candidate_documents_and_binary_scores(query_to_search, words, pls):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            for doc_id, tf in list_of_doc:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 1)
    return candidates

#search version 1 HELPER
def generate_document_binary_df(query_to_search, words, pls):
    candidates_scores = get_candidate_documents_and_binary_scores(query_to_search, words,
    pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), len(query_to_search)))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = query_to_search

    for key in candidates_scores:
        score = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = score

    return D

#HELPER - clean and tokenize a query
def tokenize_query(query):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [item for item in tokens if item not in app.all_stopwords]
    return tokens

def weightSort(docs, prw , pvw ,pr_dict , pv_dict):
    weighted = {}
    pr_max, pr_min, pr_mean = (max(pr_dict), min(pr_dict), np.mean(list(pr_dict.values())))
    pv_max, pv_min, pv_mean =(max(pv_dict), min(pv_dict), np.mean(list(pv_dict.values())))
    for doc_id in docs:
        #page rank score for document , normalized by min max
        pr = (pr_dict.get(doc_id, 1) - pr_mean) / (pr_max - pr_min)  # 1 is approximatly PR mean
        #page view score for document , normalized by minmax
        pv = (pv_dict.get(doc_id, 670) - pv_mean) / (pv_max - pv_min)  # 670 mean value of page view
        weighted[doc_id] = prw * pr + pvw * pv
    return sorted(list(weighted.items()), key=lambda x: x[1], reverse=True)

##### Evaluation
@app.route("/report")
def report():
    query = request.args.get('query', '')
    quaries_with_answers = dict(app.queries_with_answers)
    queries = list(quaries_with_answers.keys())
    scores = []
    methods = [search_version]
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

def search_body_offline(query):
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
    query = tokenize_query(query)
    # BEGIN SOLUTION
    cosine_similarity_dict = cosine_similarity(query, app.text_index)
    top_100 = get_top_n(cosine_similarity_dict, 100)
    res = [(int(id), app.id_title[id]) for id, score in top_100]
    # END SOLUTION
    return res

def search_version(query):
    # Takes 100 best candidates with cosine similarity measure.
    query = tokenize_query(query)

    body_dict = cosine_similarity(query, app.text_index)
    top_100_body = get_top_n(body_dict, 40)

    #title_dict = cosine_similarity(query, app.title_index)
    #top_100_title = get_top_n(title_dict, 100)

    #best = merge_results(top_100_body, top_100_title, scores_1_weight=0.9, scores_2_weight=0.1, N=40)
    res = [int(id) for id, score in top_100_body]
    # sort the result with pagerank and pageviews weights.
    top_pagerank = weightSort(res,prw=0.6,pvw=0.4,pr_dict=app.pagerank,pv_dict=app.wid2pv)
    return top_pagerank

def intersection(l1,l2):
    """
    This function perform an intersection between two lists.

    Parameters
    ----------
    l1: list of documents. Each element is a doc_id.
    l2: list of documents. Each element is a doc_id.

    Returns:
    ----------
    list with the intersection (without duplicates) of l1 and l2
    """
    return list(set(l1)&set(l2))

def precision_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the precision@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    return round(len(intersection(true_list, predicted_list[:k])) / k, 3)

def map_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the average_precision@k metric.(i.e., precision in every recall point).

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, average precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    relevant_indexes_k = [i + 1 for i in range(len(predicted_list[:k])) if predicted_list[i] in true_list]
    p_k_rel_k = [precision_at_k(true_list, predicted_list, i) for i in relevant_indexes_k]
    try:
        return round(sum(p_k_rel_k) / len(p_k_rel_k), 3)
    except:
        return round(0, 3)



if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

