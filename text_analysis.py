import os
import sys
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import json
import pandas as pd
from ordered_set import OrderedSet
from tqdm import tqdm
from acquisition import try_all_crypto_white_papers

nlp = spacy.load("en_core_web_sm")  # loads English library

keyword_dict = {
    "ai": ["artificial intelligence", "ai", "machine learning"],
    "music": ["music", "musician"],
    "sports": ["sport", "soccer", "football"],  # the main sports I found on cryptocompare
    "education": ["education"],
    "healthcare": ["healthcare", "hospital"],
    "tourism": ["tourism", "travel"],
    "energy": ["energy sector", "energy market", "energy industry"],  # "energy" is too vague
    "entertainment": ["entertainment"],
    "smart_contracts": ["smart contract"],
    "governance": ["governance token", "vote"],  # too many false positives w/ "governance"
    "nft": ['nft', 'non-fungible'],
    "virtual_world": ["virtual world", "virtual land"],
    "vr_ar": ["virtual reality", "vr", "augmented reality", "ar"],
    "gaming": ["video game", "gamefi", "game-fi", "gaming platform", "gaming industry"],
    "gambling": ["gamble", "bet", "casino"],
    "p2e": ["play-to-earn", "p2e"],
    "web3": ["web3"],
    "metaverse": ["metaverse"],
    "dao": ["dao", "decentralized autonomous"],
    "data_storage": ["cloud", "data storage", "decentralized storage"],
    "cybersecurity": ["cybersecurity", "cyber security"],
    "id": ["identity verification", "decentralized identity"],
    "supply_chain": ["supply chain", "logistics"],
    "e-commerce": ["e-commerce", "ecommerce"],
    "social": ["social network", "social token", "social media"],  # "social" is too vague
    "gold": ["tokenized gold", "gold bar", "physical gold"],
    "meme": ["meme", "memecoin"]
}


def text_descriptions_cc(cc_json='coin_data/cc_info.json', lim=None, truncate=True, keep_soup=False):

    with open(f"{os.getcwd()}/{cc_json}", "r") as f:
        data = json.load(f)
        df = pd.DataFrame(data).T

    text_descriptions = {}
    for i, (token, row) in enumerate(df.iterrows()):
        try:
            cc_summary = row['General']['Description']
        except TypeError:
            continue
        soup = BeautifulSoup(cc_summary, features="html.parser")  # converts text to object
        if keep_soup:
            text_descriptions[token] = soup
        else:
            text = soup.get_text()
        # If the description has a background section on the creators, then we stop reading there
            if truncate:
                text = text[:text.find("Who Created")]
            text_descriptions[token] = text
        if i+1 == lim:
            break
    return text_descriptions


def tokenize_lemma(words):
    return [w.lemma_.lower() for w in nlp(words)]


def get_tags_cc(keywords=keyword_dict):

    descriptions = text_descriptions_cc()
    stop_words_lemma = set(tokenize_lemma(' '.join(sorted(STOP_WORDS))))
    stop_words_lemma = list(stop_words_lemma.union({'-'}))
    vocab = [OrderedSet(tokenize_lemma(words)).difference(stop_words_lemma) for value in keywords.values() for words in value]
    vocab = {' '.join(kw) for kw in vocab}
    vocab_list = sorted(list(vocab))

    bow_vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), stop_words=stop_words_lemma,
                                     binary=True, tokenizer=tokenize_lemma)
    coins, categories = list(descriptions.keys()), list(keywords.keys())
    counts = bow_vectorizer.fit_transform(descriptions.values())

    #  Now, convert keyword counts to category tags for all the coins
    category_indices = {category: [] for category in categories}
    for i, (category, kws) in enumerate(keywords.items()):  # find the indices of each category's keywords...
        for kw in kws:
            processed_kw = ' '.join(OrderedSet(tokenize_lemma(kw)).difference(stop_words_lemma))
            processed_kw_i = vocab_list.index(processed_kw)
            category_indices[category].append(processed_kw_i)

    coin_tag_dict = {coin: [] for coin in coins}
    for j, category in enumerate(categories):
        category_matches = np.sum(counts[:, category_indices[category]], axis=1)  # tag if ANY keyword present
        tagged_coins = np.where(category_matches)[0]
        for coin_i in tagged_coins:  # each of these lists should be short enough that the double for-loop is fine
            coin = coins[coin_i]
            coin_tag_dict[coin].append(category)  # add the tag

    path = os.getcwd()
    with open(f'{path}\\coin_data\\cc_tags.json', 'w') as f:
        json.dump(coin_tag_dict, f)
    return coins, vocab_list, counts


def find_tag_kws_cc(coin_dict):
    descriptions = text_descriptions_cc()
    sentences_keep = {}
    inv_kw_map = {kw: tag for tag, kws in keyword_dict.items() for kw in kws}

    for coin in tqdm(coin_dict.keys(), total=len(coin_dict), desc="Searching coin descriptions for keywords...",
                     file=sys.stdout):
        sentences_keep[coin] = {}
        if coin in descriptions.keys():
            description_nlp = nlp(descriptions[coin])
            relevant_kws = [keyword_dict[tag] for tag in coin_dict[coin]["tags"] if tag in keyword_dict.keys()]
            relevant_kws = [kw for kws in relevant_kws for kw in kws]
            for kw in relevant_kws:
                kw_sents = [sent.text for sent in description_nlp.sents if kw in sent.text]
                if kw_sents:
                    sentences_keep[coin][inv_kw_map[kw]] = kw_sents[0]
    return sentences_keep


def get_white_papers(coin_dict):
    wps = try_all_crypto_white_papers(coin_dict)
    descriptions = text_descriptions_cc(keep_soup=True)

    for coin, wp_link in tqdm(wps.items(), total=len(wps), desc="Finding whitepaper links...",
                     file=sys.stdout):
        if not wp_link:
            print(f"Checking CryptoCompare file for a white paper link for {coin}...")
            if coin in descriptions.keys():
                links = descriptions[coin].find_all('a')
                for link in links:
                    if "white paper" in link.text:
                        wps[coin] = link.get('href')
    return wps


def get_first_sentences(coin_dict, n=2):
    descriptions = text_descriptions_cc()
    sentences_keep = {}
    for coin in tqdm(coin_dict.keys(), total=len(coin_dict), desc="Grabbing first sentences from description...",
                     file=sys.stdout):
        sentences_keep[coin] = {}
        if coin in descriptions.keys():
            description_nlp = nlp(descriptions[coin])
            sents = [sent.text for sent in list(description_nlp.sents)[:n]]
            sentences_keep[coin] = ' '.join(sents)
    return sentences_keep


if __name__ == "__main__":

    get_tags_cc()

