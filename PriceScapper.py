# ---------- Required Dependancys -----------
import requests
import time
import datetime as datetime
localTime = time.asctime(time.localtime(time.time()))

print(" -------------- ENTRY AT " + str(localTime) + " ---------------- ")

# ----- Price Data Scraper () -----
def ScrapePrice(CryptoName):
    # - - - Creates CSV File in Directory "Data"
    fileName = open('Data/{}_PriceData.csv'.format(CryptoName),"a")
    # - - - Columns For Price Data 
    keys = ["price_usd",
            "24h_volume_usd",
            "market_cap_usd",
            "available_supply",
            "total_supply",
            "percent_change_1h",
            "percent_change_24h",
            "percent_change_7d"]
    # - - Creates Empty list for Price Data
    vals = [0]*len(keys)
    i = 0 # - - - Loop counter
    while True:
        i += 1
        # - - - API Scraping Data from CoinMarketCap
        data = requests.get("https://api.coinmarketcap.com/v1/ticker/{}/".format(CryptoName)).json()[0]
        bkc = requests.get("https://blockchain.info/ticker").json()
        for d in data.keys():
            if d in keys:
                indx = keys.index(d)
                vals[indx] = data[d]
        for val in vals:
            fileName.write(val+",")

        #fileName.write("{},{},".format(bstamp["volume"],bstamp["vwap"]))
        fileName.write("{},{},{}".format(bkc["USD"]["sell"],bkc["USD"]["buy"],bkc["USD"]["15m"]))
        fileName.write(","+datetime.datetime.now().strftime("%y-%m-%d %H:%M"))
        fileName.write("\n")
        fileName.flush()
        print(" - - - - Itteration: " + str(i) + " - - - - ")
        # - - - Timer to loop once / min (60secs)
        time.sleep(60)

ScrapePrice("ethereum")