import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import os
import sys
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import spacy
import json
import pandas as pd
import time

nlp = spacy.load("en_core_web_sm")  # loads English library


def scrape_coin_cc(coin):

    url = f"https://www.cryptocompare.com/coins/{coin}/overview"
    params = {'api_key': ''}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code >= 300:
        print(f"The request failed for {url}... status code: {response.status_code}, reason: {response.reason}")
        return None
    else:
        soup = BeautifulSoup(response.text, features="html.parser")  # converts text to object

        try:
            x = soup.find_all('script')
            main = x[8]  # This script tag contains all of the information on each coin
        except (Exception,):
            print(f"Failed to scrape data from {url}...")
            return None

        main_text = str(main.string)
        find_dict = re.compile(r"""pageInfo.setCoinPageInfo\(({\"Response\":\"Success\",.+})""")
        match = find_dict.search(main_text)

        if match:
            coin_dict = json.loads(match.group(1))['Data']
            key_list = ['General', 'Taxonomy', 'Ratings', 'ICO']
            coin_info = {key: coin_dict[key] for key in key_list}
            return coin_info
        else:
            print(f"The request failed for {url}... An error has occured extracting the dictionary for this token...")
            return None


def get_coin_info_cc(cmc_json=r"coin_data/cmc_top2000_info.json"):

    with open(f"{os.getcwd()}/{cmc_json}", "r") as f:
        data = json.load(f)
        cmc_df = pd.DataFrame(data).T

    cc_dict = dict.fromkeys(cmc_df.index)
    for i, coin in enumerate(tqdm(cc_dict.keys(), total=len(cmc_df), desc="Scraping from CryptoCompare...",
                                  file=sys.stdout)):
        cc_info = scrape_coin_cc(coin)
        cc_dict[coin] = cc_info
        time.sleep(1.2)
        if i % 100 == 0:
            print("Saving...")
            with open(f'{os.getcwd()}/coin_data/cc_info.json', 'w') as f:
                json.dump(cc_dict, f)
    with open(f'{os.getcwd()}/coin_data/cc_info.json', 'w') as f:
        json.dump(cc_dict, f)


def get_marketcaps_cmc(cmc_json=r"coin_data/cmc_top2000_info.json", n=3000):

    with open(f"{os.getcwd()}\\{cmc_json}", "r") as f:
        data = json.load(f)
        cmc_df = pd.DataFrame(data).T
    coins = cmc_df.index.to_list()

    parameters = {
        'start': '1',
        'limit': n,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '',
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest', params=parameters)
        data = json.loads(response.text)['data']
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)
    coin_mcs = {coin['symbol']: coin['quote']['USD']['market_cap'] for coin in data}
    coin_mcs = {coin: coin_mcs[coin] for coin in coins if coin in coin_mcs.keys()}
    path = os.getcwd()
    with open(f'{path}/coin_data/cmc_marketcaps.json', 'w') as f:
        json.dump(coin_mcs, f)


def get_coin_info_cmc(n=2000):
    parameters = {
        'start': '1',
        'limit': n,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '',
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest', params=parameters)
        data = json.loads(response.text)['data']
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)
    coin_info = {coin['symbol']: coin for coin in data}
    path = os.getcwd()
    with open(f'{path}/coin_data/cmc_top{n}_info.json', 'w') as f:
        json.dump(coin_info, f)


def get_speeds_kraken():
    url = "https://support.kraken.com/hc/en-us/articles/203325283-Cryptocurrency-deposit-processing-times"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    if response.status_code >= 300:
        print(f"The request failed for {url}... status code: {response.status_code}, reason: {response.reason}")
        return  None
    else:
        soup = BeautifulSoup(response.text, features="html.parser")  # converts text to object

    speeds = {}
    table = soup.select('div.pane-vertical-scroll')[0]
    for row in table.find_all('tr'):
        in_row = row.contents

        coin_string = in_row[1].get_text()
        find_symbol = re.compile(r"""\((\w{1,6})\)""")
        match = find_symbol.search(coin_string)
        if match:
            symbol = match.group(1)
        else:
            print(f"Failed to find symbol in {coin_string}")
            continue

        speed_string = in_row[5].get_text().strip()
        if speed_string == "Near-instant":
            speed_string = "1 seconds"

        elif "-" in speed_string:
            speed_string = speed_string[speed_string.find("-")+1:]  # take the higher end of time ranges

        if "(" in speed_string:
            speed_string = speed_string[:speed_string.find("(")]  # ignore parentheses after the estimate

        for substring in ["EST", "Dependent on Fee", "~", "Under"]:  # who cares if it's an approximation...
            speed_string = speed_string.replace(substring, "")

        try:
            pd.to_timedelta(speed_string)  # Check to make sure we can convert to TimeDelta object after reading JSON
        except (Exception,):
            print(f"Speed information for {symbol} cannot be converted to a Pandas TimeDelta object...")

        speeds[symbol] = speed_string
    path = os.getcwd()
    with open(f'{path}/coin_data/kraken_speeds.json', 'w') as f:
        json.dump(speeds, f)


def get_security_ratings_cer():
    url = "https://cer.live/cryptocurrency-security-rating"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    if response.status_code >= 300:
        print(f"The request failed for {url}... status code: {response.status_code}, reason: {response.reason}")
        return None
    else:
        soup = BeautifulSoup(response.text, features="html.parser")  # converts text to object
    x = soup.find_all('script')
    find_coin_dicts = re.compile(r"""(\{\"id\":.+?(?:\"\:[0-9]+\}))""")
    matches = find_coin_dicts.findall(str(x))
    security_scores = {}
    if matches:
        for i, match in enumerate(matches):
            match = json.loads(match)
            security_scores[match["ticker"]] = (match["securityScore"], match["securityRating"])
        path = os.getcwd()
        with open(f'{path}/coin_data/CER_security_ratings.json', 'w') as f:
            json.dump(security_scores, f)
    else:
        print(f"Failed to scrape security information from {url}")
        return None


def get_energy_ratings_cw():
    url = "https://www.cryptowisser.com/crypto-carbon-footprint/"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    if response.status_code >= 300:
        print(f"The request failed for {url}... status code: {response.status_code}, reason: {response.reason}")
        return None
    else:
        soup = BeautifulSoup(response.text, features="html.parser")  # converts text to object
    ticker_cells = soup.select('td[data-label="Ticker"]')
    energy_cells = soup.select('td[data-label="Carbon Footprint"]')
    energy_scores = {}
    for ticker, energy in zip(ticker_cells, energy_cells):
        energy_scores[ticker.get_text(strip=True)] = energy.get_text(strip=True)
    path = os.getcwd()
    with open(f'{path}/coin_data/cw_energy_ratings.json', 'w') as f:
        json.dump(energy_scores, f)


def try_all_crypto_white_papers(coin_dict):
    wps = {}
    for coin in tqdm(coin_dict.keys(), total=len(coin_dict), desc="Finding whitepaper links on https://www.allcryptowhitepapers.com/...",
                     file=sys.stdout):
        name = coin_dict[coin]["name"]
        wps[coin] = {}
        url = f"https://www.allcryptowhitepapers.com/{name}-Whitepaper/"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if response.status_code >= 300:
            print(f"The request failed for {url}... status code: {response.status_code}, reason: {response.reason}")
            wps[coin] = None
        else:
            wps[coin] = url
    return wps


if __name__ == "__main__":

    # get_coin_info_cc()
    # get_marketcaps_cmc()
    get_coin_info_cmc(n=2000)
    # get_speeds_kraken()
    # get_security_ratings_cer()
    # get_energy_ratings_cw()
    # try_all_crypto_white_papers()
