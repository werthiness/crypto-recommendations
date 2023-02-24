import os
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import copy


def find_recs(tags, master_json='coin_data/feature_matrix.json', drop_rest=True, k=5):

    with open(f"{os.getcwd()}/{master_json}", "r") as f:
        features = json.load(f)
    feature_matrix = pd.DataFrame(index=features['index'], columns=features['columns'], data=features['data'])
    ranks = feature_matrix['rank']  # hold for later, but drop from feature matrix
    feature_matrix.drop(columns=['rank'], inplace=True)

    for feature in ["security", "speed", "energy_efficiency"]:
        feature_matrix[feature] = feature_matrix[feature].fillna(0)

    if drop_rest:
        feature_matrix = feature_matrix.filter(items=tags)

    user_point = pd.DataFrame(0, index=[0], columns=feature_matrix.columns)
    for tag in tags:  # add 1s to what they're looking for
        user_point[tag] = 1

    find_n = {1: 800, 2: 600, 3: 400}
    if len(tags) in find_n.keys():
        n_neighbors = find_n[len(tags)]
    else:
        n_neighbors = 100

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan', algorithm='brute').fit(feature_matrix)
    nn.fit(feature_matrix)

    distances, indices = nn.kneighbors(user_point, return_distance=True)
    coins = list(feature_matrix.index.values)
    recs = [coins[x] for x in indices[0]]

    rec_df = pd.DataFrame(index=recs, columns=['distance', 'rank'])
    for rec, distance in zip(recs, distances[0]):
        norm_distance = distance/len(feature_matrix.columns)
        rec_df.loc[rec, 'distance'] = norm_distance
        rec_df.loc[rec, 'rank'] = ranks[rec]
    rec_df.sort_values(by=['distance', 'rank'], inplace=True)

    final_recs = rec_df.index.values[:k]
    norm_distances = rec_df.distance.values.tolist()[:k]
    perc_match = [(1 - distance ) * 100 for distance in norm_distances]
    tag_values = {rec: feature_matrix.loc[rec, :].values.tolist() for rec in final_recs}
    return final_recs, perc_match, tag_values


def read_recommendation_json(final_recs, tags, rec_json='coin_data/recommendation_info.json'):

    with open(f"{os.getcwd()}/{rec_json}", "r") as f:
        rec_dict = json.load(f)
    init_sent_dict = {tag: None for tag in tags}
    init_coin_dict = {key: (copy.deepcopy(init_sent_dict) if key == "sents" else None) for key in ["name", "all_tags", "sents"]}
    highlight = {key: copy.deepcopy(init_coin_dict) for key in final_recs}
    for coin in final_recs:
        info = rec_dict[coin]
        if info['name']:
            highlight[coin]['name'] = info['name']
        highlight[coin]["symbol"] = info["symbol"]
        highlight[coin]["all_tags"] = info["tags"]
        highlight[coin]["sents"] = info["sents"]
        highlight[coin]["market_cap"] = info["market_cap"]
        highlight[coin]["white_paper"] = info["white_paper"]
        highlight[coin]["url"] = info["url"]

        if "speed" in tags:
            if "speed" in info.keys():
                speed, platform = info["speed"]
                if not platform:
                    highlight[coin]["speed"] = f"{coin} has an average transaction speed of around {speed}."
        if "energy_efficiency" in tags:
            rating_dict = {"True Green": "the most energy-efficient - carbon neutral or negative)",
                           "Medium Green": "very high energy-efficiency (as efficient as a VISA transaction, "
                                           "if not more.)",
                           "Light Green": "high energy-efficiency",
                           "Beige": "low energy-efficiency",
                           "Brown": "very low energy-efficiency"}
            if "energy_efficiency" in info.keys():
                energy_efficiency, platform = info["energy_efficiency"]
                if not platform:
                    highlight[coin]["energy_efficiency"] = f"{coin} has a rating of {energy_efficiency}, " \
                                                                    f"which means {rating_dict[energy_efficiency]}."
        if "security" in tags:
            security = info["security"]
            if "security" in info.keys():
                highlight[coin]["security"] = f"{coin} has a security rating of {security}."

        if "deflationary" in tags:
            highlight[coin]["supply"] = f"{coin} has a total supply of {info['supply']}."

    return highlight


if __name__ == "__main__":

    test_tags = ['speed', 'security', 'music']  # pick anything
    final_recs, perc_match, tag_values = find_recs(tags=test_tags)
    highlight = read_recommendation_json(final_recs, test_tags)

