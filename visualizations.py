import matplotlib.ticker
import json
import seaborn as sns
import pandas as pd
from feature_matrix import read_speed_ratings
from pylab import *
from dateutil.relativedelta import relativedelta
import os
from scipy.stats import pearsonr, gaussian_kde
import yfinance as yf
from sklearn import decomposition


sns.set_style("darkgrid", {'axes.facecolor': 'black', 'grid.color': 'black', 'axes.labelcolor': 'w',
                  'figure.facecolor': 'black', 'text-color': 'white',
                               'xtick.bottom': True, 'xtick.color': 'white', 'ytick.color': 'white'})
plt.rcParams['text.color'] = 'w'
plt.rcParams['font.family'] = "arial"

tags_to_labels = {"entertainment": "Entertainment", "meme": "Memecoins", "sports": "Sports", "social": "Social media",
                  "music": "Music",  "supply_chain": "Logistics", "e-commerce": "E-commerce", "marketing": "Marketing",
                  "insurance": "Insurance", "healthcare": "Healthcare", "tourism": "Tourism ",
                  "real_estate": "Real estate",  "energy": "Energy sector", "gold": "Tokenized gold",
                  "education": "Education",  # General
                  "gaming": "Gaming", "p2e": "Play-to-earn", "vr_ar": "VR/AR", "gambling": "Gambling",
                  "virtual_world": "Virtual land/world",  # Gaming
                  "big_data": "Big data", "ai": "Artificial intelligence", "data_storage": "Data storage",
                  "cybersecurity": "Cybersecurity", "id": "Identity verification",  # Data
                  "nft": "NFTs️", "smart_contracts": "Smart contracts", "dao": "DAOs", "metaverse": "Metaverse",
                  "web3": "Web3 ", "dex": "DEX", "stablecoin": "Stablecoins", "interoperability": "Interoperability",
                  "date": "date"}

tag_cats = {"entertainment": 0, "meme": 0, "sports": 0, "social": 0, "music": 0,  "supply_chain": 0, "e-commerce": 0,
            "marketing": 0, "insurance": 0, "healthcare": 0, "tourism": 0, "real_estate": 0,  "energy": 0, "gold": 0,
            "education": 0,  # General
            "gaming": 1, "p2e": 1, "vr_ar": 1, "gambling": 1, "virtual_world": 1,  # Gaming
            "big_data": 2, "ai": 2, "data_storage": 2, "cybersecurity": 2, "id": 2,  # Data
            "nft": 3, "smart_contracts": 3, "dao": 3, "metaverse": 3, "web3": 3, "dex": 3, "stablecoin": 3,
            "interoperability": 3}


def plot_marketcaps(market_cap_json=r"coin_data/cmc_marketcaps.json"):

    with open(f"{os.getcwd()}/{market_cap_json}", "r") as f:
        market_caps = json.load(f)
    market_caps = list(market_caps.values())
    market_caps = [x for x in market_caps if 1000000 <= x <= 1000000000000]

    bins = np.logspace(np.log10(1000000), np.log10(1000000000000), 7)
    counts, _ = np.histogram(market_caps, bins)

    sns.histplot(market_caps, log_scale=True, kde=True, ec="#ffffff", color="#217fb4")
    plt.show()


def get_dates(cc_json='coin_data/cc_info.json'):

    with open(f"{os.getcwd()}/{cc_json}", "r") as f:
        data = json.load(f)
        df = pd.DataFrame(data).T

    dates = {}
    for i, (token, row) in enumerate(df.iterrows()):
        try:
            date = row['General']['StartDate']
        except TypeError:
            continue
        try:
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue
        dates[token] = date
    return dates


def get_platforms(cmc_json='coin_data/cmc_top2000_info.json'):
    with open(f"{os.getcwd()}/{cmc_json}", "r") as f:
        cmc_data = json.load(f)
    return {coin: (f"L2_{info['platform']['symbol']}" if info["platform"] else "L1") for coin, info in cmc_data.items()}


def plot_tags_over_time(master_json='coin_data/feature_matrix.json', platform="ETH", t_step=3,
                        partial_grid=True):

    with open(f"{os.getcwd()}/{master_json}", "r") as f:
        features = json.load(f)
    df = pd.DataFrame(index=features['index'], columns=features['columns'], data=features['data'])

    for feature in ["security", "speed", "energy_efficiency"]:
        df[feature] = df[feature].fillna(0)

    dates = get_dates()
    for coin in df.index:
        if coin in dates.keys():
            df.loc[coin, "date"] = dates[coin]

    platform_start = dates[platform]
    df = df[(df[f"L2_{platform}"] == 1) & (df["date"] >= platform_start)]

    df.drop(columns=["L1", "L2_ADA", "L2_ATOM", "L2_BNB", "L2_ETH", "L2_MATIC", "L2_SOL",
                          "L2_other", "consensus_other", "pow", "pos", "dpos", "hybrid-pow-pos",
                          "cex", "deflationary", "security", "energy_efficiency", "speed", "rank"], inplace=True)
    df.sort_values(by="date", inplace=True)
    df = df[tags_to_labels.keys()]
    df.rename(tags_to_labels, axis="columns", inplace=True)

    heatmap_longform = pd.DataFrame(index=df.columns.tolist(), columns=["t_i", "count"])
    tag_zeros = {tag: [] for tag in heatmap_longform.index}
    t, t_i = platform_start, 0
    t_i_dates = {}
    t_i_projects = []
    while t.date() < datetime.date.today():
        df_t = df[df["date"] <= t]
        t_i_projects += [len(df_t)]
        data_t = df_t.sum(numeric_only=True)
        data_t = data_t.to_frame(name="count")
        data_t["t_i"] = t_i
        heatmap_longform = pd.concat([heatmap_longform, data_t])
        ti_zeros = list(data_t[data_t["count"] == 0].index)
        for tag in ti_zeros:
            tag_zeros[tag].append(t_i)
        t_i_dates[t_i] = (t.date().year, t.date().month)
        t += relativedelta(months=+t_step)
        t_i += 1
    heatmap_longform.dropna(how="any", inplace=True)
    sns.set_style("darkgrid", {'axes.facecolor': 'black', 'grid.color': 'black', 'axes.labelcolor': 'w',
                  'figure.facecolor': 'black', 'text-color': 'white',
                               'xtick.bottom': True, 'xtick.color': 'white', 'ytick.color': 'white'})
    plt.rcParams['text.color'] = 'w'
    plt.rcParams['font.family'] = "arial"

    h = sns.relplot(heatmap_longform, x="t_i", y=heatmap_longform.index, hue="count", size="count", palette="YlOrBr",
                    sizes=(0, 200), size_norm=(0, 200), hue_norm=(0, 200), height=8)
    if partial_grid:
        for tag, zeros in tag_zeros.items():
            plt.plot(zeros, [tag]*len(zeros), 'w-', label="0", lw=1)

    # Axis ticks
    ax = h.ax
    ax.set_xlim([0, t_i])  # t_i variable at maximum value at end of while loop
    year_starts = [0, 2, 6, 10, 14, 18, 22, 26]
    plt.xticks(year_starts, [t_i_dates[t_i][0] for t_i in year_starts])
    offset = matplotlib.transforms.ScaledTranslation(25/72., 0, h.figure.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels()[1:]:  # 2015 only has a few months, so no need to shift it
        label.set_transform(label.get_transform() + offset)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    # Axis label
    plt.xlabel("Year \n (Ethereum start - present)")
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    sns.move_legend(h, handles=[handles[6]]+handles[1:6], labels=[labels[6]]+labels[1:6], loc=5,
                    fontsize=9, labelcolor='w', title='# projects')
    plt.show()


def get_price_time_series(master_json='coin_data/feature_matrix.json', start='2020-01-01', end='2022-12-31'):

    with open(f"{os.getcwd()}/{master_json}", "r") as f:
        features = json.load(f)
    coins = features['index']

    price_file = f'{os.getcwd()}/coin_data/price_time-series.json'
    if os.path.isfile(price_file):
        print(f"Found price time series data already ({price_file})")
        with open(price_file, "r") as f:
            price_dict  = json.load(f)
            coin_start = len(price_dict)
    else:
        price_dict, coin_start = dict.fromkeys(coins), 0

    for i, coin in enumerate(coins[coin_start:]):
        try:
            coin_df = yf.download(tickers=f"{coin}-USD", start=start, end=end, interval='1d')
            if len(coin_df.columns) == 6:
                price_dict[coin] = list(zip(coin_df['Open'].values, coin_df['Close'].values))
        except (Exception,):
            print(f"Failed to find price time-series data for {coin}...")
            price_dict[coin] = None
        if i % 100 == 0:
            print("Saving...")
            with open(price_file, 'w') as f:
                json.dump(price_dict, f)
    with open(price_file, 'w') as f:
        json.dump(price_dict, f)


def write_btc_correlation(price_json="coin_data/price_time-series.json", daily_metric='Close'):

    with open(f"{os.getcwd()}/{price_json}", "r") as f:
        prices = json.load(f)
    btc_changes = [(c - o)/o for (o, c) in prices["BTC"]]

    correlation_dict = {}
    for coin in prices.keys():
        coin_changes = [(c - o)/o for (o, c) in prices[coin] if o != 0]
        if len(coin_changes) >= 30:  # at least a month of data; arbitrary
            correlation_dict[coin] = pearsonr(btc_changes[:len(coin_changes)], coin_changes)

    with open(f'{os.getcwd()}\\coin_data\\price_btc_correlations.json', 'w') as f:
        json.dump(correlation_dict, f)


def plot_btc_correlation(corr_json="coin_data/price_btc_correlations.json",
                         master_json='coin_data/feature_matrix.json'):

    with open(f"{os.getcwd()}/{corr_json}", "r") as f1:
        correlations = json.load(f1)

    with open(f"{os.getcwd()}/{master_json}", "r") as f2:
        features = json.load(f2)
    feature_matrix = pd.DataFrame(index=features['index'], columns=features['columns'], data=features['data'])
    feature_matrix.drop(columns=["L1", "L2_ADA", "L2_ATOM", "L2_BNB", "L2_ETH", "L2_MATIC", "L2_SOL",
                          "L2_other", "consensus_other", "pow", "pos", "dpos", "hybrid-pow-pos",
                          "cex", "deflationary", "security", "energy_efficiency", "speed", "rank", "governance",
                          "privacy"], inplace=True)

    longform_correlation = pd.DataFrame(columns=["category", "correlation"])
    means = {}
    for tag in feature_matrix.columns:
        tagged_coins = feature_matrix[feature_matrix[tag] == 1].index.tolist()
        tagged_coin_correlations = [correlations[coin][0] for coin in tagged_coins if coin in correlations.keys()]
        if len(tagged_coin_correlations) >= 2:
            tagged_coin_cats = [tag] * len(tagged_coin_correlations)
            tag_longform = pd.DataFrame(columns=["category", "correlation"])
            tag_longform["category"] = tagged_coin_cats
            tag_longform["correlation"] = tagged_coin_correlations
            longform_correlation = pd.concat([longform_correlation, tag_longform])
            means[tag] = np.mean(tagged_coin_correlations)
    # Sort by means
    means = pd.DataFrame(means.values(), index=means.keys(), columns=["mean"])
    means["tag_cat"] = pd.Series(tag_cats)
    means.sort_values(["tag_cat", "mean"], ascending=[True, False], axis=0, inplace=True)
    longform_correlation.sort_values('category', inplace=True,
                                     key=lambda col: col.apply(lambda x: means.index.get_loc(x)))

    # Plot KDEs
    g = sns.FacetGrid(longform_correlation, row="category", hue="category", aspect=10, height=.5)
    g.map(sns.kdeplot, "correlation",
          common_norm=False, bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1, zorder=1, cut=0)
    # g.map(sns.kdeplot, "correlation", clip_on=False, color=None, lw=2, bw_adjust=.5, zorder=1, cut=0)

    # Plot category means
    for j, ax in enumerate(g.axes_dict.values()):
        ax.plot(means["mean"][j], [0], "w^", zorder=2)
        ax.axline((0, 0), slope=0, c=".2", ls="--", zorder=0)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "category")
    g.figure.subplots_adjust(hspace=0)
    g.set_titles("")
    g.set(xlabel="", xlim=[-1, 1], xticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
          ylabel="", yticks=[])
    g.despine(bottom=True, left=True)
    plt.show()


def plot_speed_scaling():
    log, lin = read_speed_ratings(return_original=True, extrapolate=False)
    lin = [pd.to_timedelta(speed[0]).total_seconds() for speed in lin.values()]

    lin = [1.-x/max(lin) for x in lin]
    lin_longform = pd.DataFrame(data=lin, index=range(len(lin)), columns=["value"])
    lin_longform["scale"] = "Linear"
    log_longform = pd.DataFrame(data=log.values(), index=range(len(log.values())), columns=["value"])
    log_longform["scale"] = "Logarithmic"
    df = pd.concat([lin_longform, log_longform], ignore_index=True)
    rcParams['figure.figsize'] = 2.5, 2.5
    ax = sns.kdeplot(df, x="value", hue="scale", cut=0, bw_adjust=2)
    sns.move_legend(ax, "upper left")
    plt.xlabel("Speed rating")
    plt.ylabel("Density")
    plt.xlim([0, 1])
    plt.yticks([0, 1, 2, 3, 4])
    plt.show()


def plot_pca_space(master_json='coin_data/feature_matrix.json', all_plots=True, cmap='jet'):

    with open(f"{os.getcwd()}/{master_json}", "r") as f:
        features = pd.read_json(f, orient='split')
    features.drop(columns=["rank"], inplace=True)
    # original_dimensions = features.columns
    feature_matrix = features.to_numpy()
    feature_matrix = np.nan_to_num(feature_matrix).astype(float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    pca = decomposition.PCA(n_components=3)
    pca_matrix = pca.fit_transform(feature_matrix)
    print(pca.explained_variance_ratio_)
    densObj = gaussian_kde(pca_matrix.T)

    def colorize(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colors = [cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(val) for val in vals]
        return colors
    colors = colorize(densObj.evaluate(pca_matrix.T))

    ax.scatter(pca_matrix[:, 0], pca_matrix[:, 1], pca_matrix[:, 2],
               alpha=0.4, color=colors, depthshade=False)

    ax.set_xlabel('PC1', fontsize=18)
    ax.set_ylabel('PC2', fontsize=18)
    ax.set_zlabel('PC3', fontsize=18)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    rcParams['figure.figsize'] = 6, 6
    plt.show()

    if all_plots:
        labels = ['PC1', 'PC2', 'PC3']
        for (d1, d2) in [(0, 1), (0, 2), (1, 2)]:
            two_pcs = np.vstack([pca_matrix[:, d1], pca_matrix[:, d2]])
            # densObj = gaussian_kde(two_pcs)
            # colors = colorize(densObj.evaluate(two_pcs))
            # ax.scatter(two_pcs[0, :], two_pcs[1, :], alpha=0.4, color=colors)
            g = sns.JointGrid(x=two_pcs[0, :], y=two_pcs[1, :], space=0)
            g.fig.set_size_inches((4, 4))
            g.plot_joint(sns.kdeplot,
                         fill=True, thresh=0, levels=100, cmap=cmap, ax=ax)
            g.plot_marginals(sns.kdeplot, color="#ffffff", alpha=1)
            g.set_axis_labels(xlabel=labels[d1], ylabel=labels[d2], fontsize=18)
            g.ax_joint.set_xticks([])
            g.ax_joint.set_yticks([])
            plt.show()


if __name__ == "__main__":

    # plot_marketcaps()
    # plot_tags_over_time()
    # plot_speed_scaling()
    plot_pca_space(cmap='rainbow')
