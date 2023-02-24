import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import json
import pandas as pd
from text_analysis import get_first_sentences, get_white_papers


def read_tags_cmc(cmc_json='coin_data/cmc_top2000_info.json'):

    category_conversions = {
        "ai-big-data": "big_data",
        "centralized-exchange": "cex", "collectibles-nfts": "nft",
        "cybersecurity": "cybersecurity", "dao": "dao", "decentralized-exchange-dex-token": "dex",
        "e-commerce": "e-commerce", "education": "education", "energy": "energy",
        "entertainment": "entertainment", "gaming": "gaming",
        "gambling": "gambling", "governance": "governance", "health": "healthcare", "identity": "id",
        "insurance": "insurance", "interoperability": "interoperability", "logistics": "supply_chain",
        "marketing": "marketing", "memes": "meme", "metaverse": "metaverse", "music": "music",
        "play-to-earn": "p2e", "privacy": "privacy",
        "real-estate": "real_estate", "smart-contracts": "smart_contracts", "social-money": "social",
        "sports": "sports", "stablecoin": "stablecoin", "storage": "data_storage",
        "tokenized-gold": "gold", "tourism": "tourism",
        "vr-ar": "vr_ar", "web3": "web3"
    }
    consensus = ["pow", "pos", "poa", "poc", "poi", "pop", "posign", "post", "powt", "dpos",
                 "hybrid-pow-pos", "hybrid-dpow-pow", "hybrid-pow-npos"]

    with open(f"{os.getcwd()}/{cmc_json}", "r") as f:
        data = json.load(f)

    cmc_tags = {coin: data[coin]['tags'] for coin in data.keys()}

    cmc_layer_tags = {coin: ([f"L2_{info['platform']['symbol']}"] if info["platform"] else ["L1"]) for coin, info in
                      data.items()}  # Layer 1 tokens shouldn't have a platform listed

    cmc_category_tags = dict.fromkeys(list(cmc_tags.keys()))
    cmc_consensus_tags = dict.fromkeys(list(cmc_tags.keys()))
    for coin in cmc_tags.keys():
        cmc_category_tags[coin] = [category_conversions[tag] for tag in cmc_tags[coin] if tag in
                                   category_conversions.keys()]
        cmc_consensus_tags[coin] = [tag for tag in cmc_tags[coin] if tag in consensus]

        if not cmc_consensus_tags[coin]:
            if cmc_layer_tags[coin] and (cmc_layer_tags[coin][:2] == 'L2'):
                platform = cmc_layer_tags[coin][3:]
                if platform in cmc_consensus_tags.keys():
                    cmc_consensus_tags[coin] = cmc_consensus_tags[platform]

    cmc_ranks = {coin: info['cmc_rank'] for coin, info in data.items()}

    return cmc_category_tags, cmc_consensus_tags, cmc_layer_tags, cmc_ranks


def read_tags_cc(cc_tag_json='coin_data/cc_tags.json'):
    with open(f"{os.getcwd()}/{cc_tag_json}", "r") as f:
        return json.load(f)


def fit_tags_to_site_categories(tag_dict, tags_on_site, replacement="other"):
    for k, v_list in tag_dict.items():
        for i, v in enumerate(v_list):
            if v not in tags_on_site:
                tag_dict[k][i] = replacement
    return tag_dict


def combine_tag_dicts(dict1, dict2):
    return {k: list(set(dict1.get(k, []) + dict2.get(k, []))) for k in set(list(dict1.keys()) + list(dict2.keys()))}


def read_speed_ratings(speed_json='coin_data/kraken_speeds.json', extrapolate=True, return_original=False):
    with open(f"{os.getcwd()}/{speed_json}", "r") as f:
        speed_data = json.load(f)
    original = {coin: (speed, None) for coin, speed in speed_data.items()}

    txn_speeds = {coin: pd.to_timedelta(speed).total_seconds() for coin, speed in speed_data.items()}
    # Assigning a 0-1 rating based on logarithmic assessment of transaction speeds
    min_speed, max_speed = min(txn_speeds.values()), max(txn_speeds.values())
    log_space_speeds = np.logspace(np.log10(max_speed), np.log10(min_speed), 100).tolist()  # log space of txn speed
    lin_space_ratings = np.linspace(0.01, 1.0, 100).tolist()  # the possible ratings, from 0.1 to 100

    # Not easily readable, but this finds the nearest point in the log space of txn speed, and its corresponding rating
    convert_txn_speed_to_rating = {k: v for i, (k, v) in enumerate(zip(log_space_speeds, lin_space_ratings))}
    speeds = dict.fromkeys(txn_speeds.keys())
    for coin, txn_speed in txn_speeds.items():
        nearest_speed = log_space_speeds[min(range(len(log_space_speeds)),
                                             key=lambda i: abs(log_space_speeds[i] - txn_speed))]
        speeds[coin] = convert_txn_speed_to_rating[nearest_speed]

    if extrapolate:
        _, _, layer_tags, _ = read_tags_cmc()
        for coin in layer_tags.keys():
            if coin not in speeds.keys():
                layer_tag = layer_tags[coin]
                if layer_tag[0][:2] == "L2":
                    platform = layer_tag[0][3:]
                    if platform in speeds.keys():
                        speeds[coin] = speeds[platform]
                        original[coin] = (original[platform][0], platform)
    if return_original:
        return speeds, original
    else:
        return speeds


def read_security_ratings(security_json='coin_data/CER_security_ratings.json', normalize=True, return_original=False):
    with open(f"{os.getcwd()}/{security_json}", "r") as f:
        security_data = json.load(f)
    security = {coin: (security_data[coin][0] if isinstance(security_data[coin][0], (int, float)) else np.nan)
                for coin in security_data.keys()}
    if normalize:
        max_rating = max(security.values())
        security = {coin: security[coin]/max_rating for coin in security.keys()}
    if return_original:
        original = {coin: security_data[coin][1] for coin in security_data.keys()}
        return security, original
    else:
        return security


def read_energy_ratings(energy_json='coin_data/cw_energy_ratings.json', extrapolate=True, return_original=False):
    with open(f"{os.getcwd()}/{energy_json}", "r") as f:
        energy_data = json.load(f)
    quantify = {'Brown': 0, 'Beige': 0.25, 'Light Green': 0.5, 'Medium Green': 0.75, 'True Green': 1., 'N/A': np.nan}
    energy = {coin: quantify[energy_rating] for coin, energy_rating in energy_data.items()}
    original = {coin: (rating, None) for coin, rating in energy_data.items()}

    if extrapolate:
        _, _, layer_tags, _ = read_tags_cmc()
        for coin in layer_tags.keys():
            if coin not in energy.keys() or energy[coin] == np.nan:
                layer_tag = layer_tags[coin]
                if layer_tag[0][:2] == "L2":
                    platform = layer_tag[0][3:]
                    if platform in energy.keys() and energy[platform]:
                        energy[coin] = energy[platform]
                        original[coin] = (original[platform][0], platform)
    if return_original:
        return energy, original
    else:
        return energy


def find_deflationary_coins(cc_json='coin_data/cc_info.json', cmc_json='coin_data/cmc_top2000_info.json',
                            return_supply=False):

    with open(f"{os.getcwd()}/{cmc_json}", "r") as f:
        cmc_data = json.load(f)
    deflationary = dict.fromkeys(cmc_data.keys())
    supply = dict.fromkeys(cmc_data.keys())

    for coin, info in cmc_data.items():
        if info['max_supply']:
            deflationary[coin] = 1
            supply[coin] = info['max_supply']
        else:
            deflationary[coin] = 0
            info['max_supply'] = -1

    with open(f"{os.getcwd()}/{cc_json}", "r") as f:
        cc_data = json.load(f)
        cc_df = pd.DataFrame(cc_data).T
    for coin, info in cc_df.iterrows():
        general_info = info['General']
        if general_info and ('TotalCoinSupply' in general_info.keys()):
            if general_info['TotalCoinSupply'] > 0:
                deflationary[coin] = 1
                supply[coin] = general_info['TotalCoinSupply']

    if return_supply:
        return deflationary, supply
    else:
        return deflationary


def build_feature_matrix():

    cc_tags = read_tags_cc()
    cmc_category_tags, cmc_consensus_tags, cmc_layer_tags, cmc_ranks = read_tags_cmc()
    consensus_final = ["pow", "pos", "hybrid-pow-pos", "dpos"]
    cmc_consensus_tags = fit_tags_to_site_categories(cmc_consensus_tags, consensus_final, "consensus_other")

    layer_final = ["L1", "L2_ADA", "L2_ATOM", "L2_BNB", "L2_ETH", "L2_LUNA", "L2_MATIC", "L2_SOL"]
    cmc_layer_tags = fit_tags_to_site_categories(cmc_layer_tags, layer_final, "L2_other")

    category_tags = combine_tag_dicts(cc_tags, cmc_category_tags)
    network_tags = combine_tag_dicts(cmc_layer_tags, cmc_consensus_tags)
    all_tags = combine_tag_dicts(category_tags, network_tags)

    mlb = MultiLabelBinarizer()
    binary_data = mlb.fit_transform(all_tags.values())
    master_df = pd.DataFrame(index=all_tags.keys(), columns=mlb.classes_, data=binary_data)  # start w/ binary data
    master_df['speed'] = pd.Series(read_speed_ratings())
    master_df['security'] = pd.Series(read_security_ratings())
    master_df['energy_efficiency'] = pd.Series(read_energy_ratings())
    master_df['deflationary'] = pd.Series(find_deflationary_coins())
    master_df['rank'] = pd.Series(cmc_ranks)
    master_df.sort_values(by=['rank'], inplace=True)

    path = os.getcwd()
    master_df.to_json(f'{path}/coin_data/feature_matrix.json', orient="split")
    print(f"Saved master DataFrame of tags to '{path}/coin_data/feature_matrix.json'")


def get_project_urls(cc_json='coin_data/cc_info.json'):

    with open(f"{os.getcwd()}/{cc_json}", "r") as f:
        data = json.load(f)
        df = pd.DataFrame(data).T

    urls = {}
    for i, (token, row) in enumerate(df.iterrows()):
        try:
            urls[token] = row['General']['WebsiteUrl']
        except TypeError:
            continue
    return urls


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(MyEncoder, self).default(obj)


def create_recommendation_json(master_json='coin_data/feature_matrix.json',
                               cmc_json='coin_data/cmc_top2000_info.json'):

    with open(f"{os.getcwd()}/{master_json}", "r") as f:
        features = json.load(f)
    feature_matrix = pd.DataFrame(index=features['index'], columns=features['columns'], data=features['data'])
    ranks = feature_matrix['rank']  # hold for later, but drop from feature matrix
    feature_matrix.drop(columns=['rank'], inplace=True)

    rec_dict = dict.fromkeys(feature_matrix.index)
    for coin, features in feature_matrix.iterrows():
        tagged_features = features[features == 1].index.tolist()
        rec_dict[coin] = {"tags": tagged_features, "rank": ranks[coin]}

    with open(f"{os.getcwd()}/{cmc_json}", "r") as f:
        cmc_data = json.load(f)
    consensus = ["pow", "pos", "poa", "poc", "poi", "pop", "posign", "post", "powt", "dpos",
                 "hybrid-pow-pos", "hybrid-dpow-pow", "hybrid-pow-npos"]
    for coin, info in cmc_data.items():
        rec_dict[coin]['name'] = info['name']
        rec_dict[coin]['symbol'] = info['symbol']
        # adding consensus information explicitly, since some will be tagged as "other"
        rec_dict[coin]['consensus'] = [tag for tag in info['tags'] if tag in consensus]
        rec_dict[coin]['market_cap'] = info['quote']['USD']['market_cap']

    _, original_speeds = read_speed_ratings(return_original=True)
    for coin, speed in original_speeds.items():
        if coin in rec_dict.keys():
            rec_dict[coin]["speed"] = speed

    _, original_securities = read_security_ratings(return_original=True)
    for coin, security in original_securities.items():
        if coin in rec_dict.keys():
            rec_dict[coin]["security"] = security

    _, original_energy_ratings = read_energy_ratings(return_original=True)
    for coin, energy in original_energy_ratings.items():
        if coin in rec_dict.keys():
            rec_dict[coin]["energy_efficiency"] = energy

    _, supplies = find_deflationary_coins(return_supply=True)
    for coin, supply in supplies.items():
        if coin in rec_dict.keys():
            rec_dict[coin]["supply"] = supply

    tag_sents = get_first_sentences(rec_dict)
    for coin, sents in tag_sents.items():
        rec_dict[coin]["sents"] = sents

    wps = get_white_papers(rec_dict)
    for coin, link in wps.items():
        rec_dict[coin]["white_paper"] = link

    project_urls = get_project_urls()
    for coin, url in project_urls.items():
        if coin in rec_dict.keys():
            rec_dict[coin]["url"] = url

    with open(f'{os.getcwd()}/coin_data/recommendation_info.json', 'w') as f:
        json.dump(rec_dict, f, cls=MyEncoder)


if __name__ == "__main__":

    # build_feature_matrix()
    # with open(f"{os.getcwd()}/coin_data/feature_matrix.json", "r") as f:
    #     features = json.load(f)
    # feature_matrix = pd.DataFrame(index=features['index'], columns=features['columns'], data=features['data'])

    create_recommendation_json()
