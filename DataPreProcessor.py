#region =========   Dependancys & Data    ==========

# - Data Modeling
import pandas as pd
from pandas_datareader import data as pd_data
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pytz
import seaborn as sb; sb.set()

# - - Visualisation
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.pyplot import figure
from matplotlib.legend import Legend

# - - - Saving & loading
import csv
import pickle


import time
from datetime import datetime
localTime = time.asctime(time.localtime(time.time()))
print(" -------------- DataPreProcessor - ENTRY AT " + str(localTime) + " ---------------- ")
#endregion ====== Dependancys & Data  ==============
#region ----------- Global Variables ---------------
My_DPI = 400
# Ethereum Columns Name
ETH_Original_Columns = [ "price_usd",
                         "24h_volume_usd",
                         "market_cap_usd",
                         "available_supply",
                         "total_supply",
                         "percent_change_1h",
                         "percent_change_24h",
                         "percent_change_7d",
                         "Time_and_Date"]

Price_Columns = ["price_usd",
                "24h_volume_usd",
                "Time_and_Date"]

Sent_Columns = ["sentiment", 
                "Time_and_Date"]

# - - Pickled Data Paths In Heirachial Order
Pkl_Paths = "Data/Pickled_Data/"
TmCs_Path = "TimeContinuousData/"
No_NA_Path = "NoNA_Data/"


# - DeDuplicated Dataframes, with datetime index
ETHpriceDF = pd.read_pickle(Pkl_Paths + str("NoDupe_ETH_3xTrimd_PriceData.pkl"))
MLsentDF = pd.read_pickle(Pkl_Paths + str("NoDupe_TweetMLSent_Ethereum&Crypto&bitcoin.pkl"))
LXsentDF = pd.read_pickle(Pkl_Paths + str("NoDupe_TweetLXSent_Ethereum&Crypto&bitcoin.pkl"))


# - - Continous date Time index, No Duplicates, Pandas Dataframe
TmCs_PriceDF = pd.read_pickle(Pkl_Paths + TmCs_Path + str( "ETHPriceDataNoDuplicates.pkl"))
TmCs_MLSntDF = pd.read_pickle(Pkl_Paths + TmCs_Path + str( "MLSentDataNoDuplicates.pkl"))
TmCs_LXSntDF = pd.read_pickle(Pkl_Paths + TmCs_Path + str( "LXSentDataNoDuplicates.pkl"))


# - - - Imputed continuous Pandas Dataframes
Intrp_PriceDF = pd.read_pickle(Pkl_Paths + TmCs_Path + No_NA_Path +
                 str( "interp_PriceDF.pkl"))

mean_MLSntDF = pd.read_pickle(Pkl_Paths + TmCs_Path + No_NA_Path +
                 str( "mean_MLSentDF.pkl"))

mean_LXSntDF = pd.read_pickle(Pkl_Paths + TmCs_Path + No_NA_Path +
                 str( "mean_LXSentDF.pkl"))

# - - - - Colour Schemes
MLColor_Palette =  sb.color_palette("YlOrRd_d")
LXColor_Palette =  sb.color_palette("GnBu_d")
AcColor_Palette =  sb.color_palette("Greys_d")


Clr_Palete = pd.DataFrame({"Actual": AcColor_Palette, 
                           "LXSent": LXColor_Palette, 
                           "MLSent": MLColor_Palette})

Snt_Clr_Palete = pd.DataFrame({"price_usd": AcColor_Palette,
                               "24h_volume_usd": AcColor_Palette,
                               "LXSent": LXColor_Palette, 
                               "MLSent": MLColor_Palette})


#endregion



def Clean_and_SaveData(fileName, Columns):
    #region ----- Importing & Organising Data ----
    # - Opening CSV into Panda DF from Directory
    Data_path = "Data/Collected_Data/"
    df = pd.read_csv(str(Data_path) + fileName + ".csv")
    # - - Assigning columns
    df.columns = Columns
    # - - - Converting date string to Pandas Datetime obj
    df["Time_and_Date"] = pd.to_datetime(df["Time_and_Date"], 
                                            dayfirst = False, 
                                            yearfirst = True)
    # - - - - Dropping Duplicates
    df = df.drop_duplicates(["Time_and_Date"], keep = 'first')
    # - - - - Setting Date as index and dropping its column
    df.index = df["Time_and_Date"]
    df = df.drop(["Time_and_Date"], axis = 1)
    #endregion
    #region ----- Saving Data as Pickle ----------
    New_Path = "Data/Pickled_Data/"
    df.to_pickle(str(New_Path) + "NoDupe_" + fileName +".pkl")
    #endregion
    #region ------- Saving Data as CSV -----------
    New_Path = "Data/Conditioned_Data/"
    df.to_csv(str(New_Path) + "NoDupe_" + fileName +".csv")
    #endregion
    #region ----- Displaying Finihsed DF ---------
    print("Data Frame: " + str(fileName))
    print(df)
    #endregion
    return df
# Clean_and_SaveData("ETH_3xTrimd_PriceData", Price_Columns)
# Clean_and_SaveData("TweetLXSent_Ethereum&Crypto&bitcoin", Sent_Columns)
# Clean_and_SaveData("TweetMLSent_Ethereum&Crypto&bitcoin", Sent_Columns)


# (DF, ColName<string>, Avg/Min<int>, Avg/Hour<int> *2)
def MultiPlot(DF, col, Tsample1, Tsample2, Tsample3):
    #region -------- Generating Title ------------
    print("*** Title already begins with 'Graph to show ' ***")
    title = input("Enter Desired Graph Title:")
    GraphName = title
    GraphName = GraphName.replace(" ", "_")
    #endregion
    #region -------- Plot Dimensions -------------
    figure( num=None, 
            figsize=(8, 4.5), 
            facecolor='w', 
            edgecolor='k')
    #endregion
    #region ---------- Plot Data -----------------
    DF[col].resample('{}T'.format(Tsample1)).mean().plot(alpha=0.5, style='-')
    DF[col].resample('{}H'.format(Tsample2)).mean().plot(style=':')
    DF[col].resample('{}H'.format(Tsample3)).mean().plot(style='--')
    #endregion
    #region -------- Plot Asthetics --------------
    # - Legend; position, size, visuals
    plt.legend(['Average / {} Miniute'.format(Tsample1), 
                'Average / {} Hours'.format(Tsample2), 
                'Average / {} Hours'.format(Tsample3)],
    loc='lower right')
    # - - Graph Title
    plt.title("Graph to show {}".format(title))
    # - - - Puts "$" infront of Y axis
    if col == "price_usd":
        tick = mtick.FormatStrFormatter('$%.0f')
        ax = plt.axes()
        ax.yaxis.set_major_formatter(tick)
    # - - - - Reduces White Boarder Size
    plt.tight_layout()
    #endregion
    #region --------- Saving Plot  ---------------
    plt.savefig("Graphs/Multi_Plots/NEW_Multi_Plots/{}_avg{}&_avg{}&_avg{}.png".format(GraphName, 
                                                                Tsample1,
                                                                Tsample2,
                                                                Tsample3), 
                                                          dpi = My_DPI)
    plt.show()
    #endregion

# SampleDataSet = pd.read_pickle("Graphs/Predictions/CompoundPlots/SampleDataSet.pkl")

def MultiPlot3(DF, Tsample1):
    #region -------- Generating Title ------------
    print("*** Title already begins with 'Graph to show ' ***")
    title = input("Enter Desired Graph Title:")
    GraphName = title
    GraphName = GraphName.replace(" ", "_")
    #endregion
    #region -------- Plot Dimensions -------------
    fig, ax = plt.subplots( num=None, 
                            figsize=(8, 3.5), 
                            facecolor='w', 
                            edgecolor='k')
    #endregion
    #region ---------- Plot Data -----------------
    # - empty lists for Styles & Data
    lines = []
    MeanLines = []
    Collumns = []
    styles = ['-', '--']
    for col in DF.columns:
        Collumns.append(col)
        # - Solid Lines Styling and Data
        lines  +=  ax.plot( DF[col].resample("2T").mean(), 
                          styles[0], 
                          alpha = 0.7 , 
                          color = Clr_Palete[col][2],
                          linewidth = 0.75)

        # - Average lines Styling and Data 
        MeanLines += ax.plot( DF[col].resample('{}H'.format(Tsample1)).mean(), 
                              styles[1], 
                              color = Clr_Palete[col][4],
                              linewidth = 1.5)
    #endregion      
    #region ---------- Plot Legend ---------------
    # specify the lines and labels of the first legend
    ax.legend(lines, 
             Collumns,
             loc='upper right', 
             frameon=False)
    # Create the second legend and add the artist manually.
    Collumns = ['{} / {} Hours'.format(element,Tsample1) for element in Collumns]
    leg = Legend(ax, MeanLines, 
                Collumns,
                loc='upper left', 
                frameon=False)
    ax.add_artist(leg)
    #endregion
    #region ------- Plot Periferals --------------
    # - - Graph Title
    plt.title("Graph to show {}".format(title))
    # - - - Puts "$" infront of Y axis
    tick = mtick.FormatStrFormatter('$%.0f')
    ax.yaxis.set_major_formatter(tick)
    # - - - - Reduces White Boarder Size
    plt.tight_layout()
    #endregion
    #region --------- Saving Plot  ---------------
    plt.savefig("Graphs/Predictions/CompoundPlots/{}_avg{}.png".format(GraphName, 
                                                                       Tsample1), 
                                                                dpi = My_DPI)
    #endregion
    plt.show()

def MultiSentPlot(DF, Tsample1, Tsample2):
  #region -------- Generating Title ------------
    print("*** Title already begins with 'Graph to show ' ***")
    title = input("Enter Desired Graph Title:")
    GraphName = title
    GraphName = GraphName.replace(" ", "_")
    #endregion
    #region -------- Plot Dimensions -------------
    fig, ax = plt.subplots( num=None, 
                            figsize=(8, 3.5), 
                            facecolor='w', 
                            edgecolor='k')
    #endregion
    #region ---------- Plot Data -----------------
    # - empty lists for Styles & Data
    lines = []
    MeanLines = []
    Collumns = []
    styles = ['-', '--',":"]
    for col in DF.columns:
        Collumns.append(col)
        # - Solid Lines Styling and Data
        if col == "LXSent" or col == "MLSent":
            lines  +=  ax.plot( DF[col].resample("5T").mean(), 
                            styles[0], 
                            alpha = 0.9 , 
                            color = Snt_Clr_Palete[col][2],
                            linewidth = 1)
        else:
            lines  +=  ax.plot( DF[col].resample("2T").mean(), 
                            styles[0], 
                            alpha = 0.7 , 
                            color = Snt_Clr_Palete[col][2],
                            linewidth = 0.75)

        # - Average lines Styling and Data 
        MeanLines += ax.plot( DF[col].resample('{}H'.format(Tsample1)).mean(), 
                              styles[1], 
                              color = Snt_Clr_Palete[col][4],
                              linewidth = 1.5)
    #endregion      
    #region ---------- Plot Legend ---------------
    # specify the lines and labels of the first legend
    ax.legend(lines, 
             Collumns,
             loc='upper right', 
             frameon=False)
    # Create the second legend and add the artist manually.
    Collumns = ['{} / {} Hours'.format(element,Tsample1) for element in Collumns]
    leg = Legend(ax, MeanLines, 
                Collumns,
                loc='upper left', 
                frameon=False)
    ax.add_artist(leg)
    #endregion
    #region ------- Plot Periferals --------------
    # - - Graph Title
    plt.title("Graph to show {}".format(title))
    # - - - Puts "$" infront of Y axis
    # tick = mtick.FormatStrFormatter('$%.0f')
    # ax.yaxis.set_major_formatter(tick)
    # - - - - Reduces White Boarder Size
    plt.tight_layout()
    #endregion
    #region --------- Saving Plot  ---------------
    plt.savefig("Graphs/Predictions/CompoundPlots/{}_avg{}.png".format(GraphName, 
                                                                       Tsample1), 
                                                                dpi = My_DPI)
    #endregion
    plt.show()




def Data_4_Sent_Graph(MLDataNm, LXDataNm, DataFraq):
    #region ---------- Loading PD data -----------------
    # - Loading LXSent MLSent and Price from Pickle Files
    Prc_LXSent_DF = pd.read_pickle( Pkl_Paths + 
                                    str("6DataSets/") + 
                                    LXDataNm + 
                                    str(".pkl"))

    Prc_MLSent_DF = pd.read_pickle( Pkl_Paths + 
                                    str("6DataSets/") + 
                                    MLDataNm + 
                                    str(".pkl"))
    #endregion
    #region ------ Trimming & Organising DF ------------
    # - Renaming Coloumns
    Prc_LXSent_DF.columns = ["price_usd","24h_volume_usd", "LXSent"]
    Prc_MLSent_DF.columns = ["price_usd","24h_volume_usd", "MLSent"]

    # - - Merging the two pandas data frames
    merged_DF = Prc_LXSent_DF.merge(Prc_MLSent_DF,
                            left_index = True,
                            right_index = True,
                            how = 'inner')
    # - - - Selecting Neccry columns
    print(merged_DF)
    merged_DF = merged_DF[["LXSent","MLSent","price_usd_x"]]
    # - - - - Renaming Collumns
    merged_DF.columns = ["LXSent","MLSent","price_usd"]

    # - - - - - Selecting Rows to Output
    Start_Point =  int(len(merged_DF) * DataFraq)
    merged_DF = merged_DF.ix[Start_Point:, merged_DF.columns]

    # - - - - - Normalising data frame
    merged_DF = (merged_DF-merged_DF.min())/(merged_DF.max()-merged_DF.min())
    print(merged_DF)
    #endregion
    return merged_DF

# Data_4_Sent_Graph("dropML_ETH_DF","dropLX_ETH_DF", 0.8)


# Data_4_Sent_Graph("dropML_ETH_DF","dropLX_ETH_DF",0.8)


# MultiSentPlot(Data_4_Sent_Graph("dropML_ETH_DF","dropLX_ETH_DF",0.8), 2, 30)

# print(Clr_Palete.shape)

# print(Clr_Palete.ix[:,1][2])

# print(' ----------- Devide ------------ ')

# print(Clr_Palete["Actual"][2])





# Temp_DF = pd.read_pickle(Pkl_Paths + "6DataSets/" + str( "dropLX_ETH_DF.pkl"))
# Temp_DF.to_csv(Pkl_Paths + "6DataSets/" + "Dropped_NAs/" + str( "dropML_ETH_DF.csv"))
# print(len(Temp_DF))



# MultiPlot(Temp_DF,"price_usd", 1, 6 , 12)

# MultiPlot(LXsentDF,"sentiment", 60, 6 , 12)

# MultiPlot(MLsentDF,"sentiment", 60, 6 , 12)



# LX_ETH_DF = pd.read_pickle(Pkl_Paths + "6DataSets/" + str( "LX_ETH.pkl"))
# ML_ETH_DF = pd.read_pickle(Pkl_Paths + "6DataSets/" + str( "ML_ETH.pkl"))




# TimeFiller(MLsentDF, "T", True)
def TimeFiller(DF, freq, save):
    #region ---- Creating Datetime series --------
    print(" - - - -  Original Length: " + str(len(DF)) + " - - - - ")
    new_idx = pd.date_range(start = min(DF.index), 
                            end   = max(DF.index), 
                            freq  = freq)
    # - Reindexing DF, implacing NAN where empty
    ReIndexDF = DF.reindex(new_idx)
    print(" - - - - ReIndexed Length: "+ str(len(ReIndexDF)) + " - - - - ")
    #endregion
    #region ----- Saving Data as Pickle  ---------
    if save == True:
        New_Path = "Data/Pickled_Data/TimeContinuousData/"
        fileName = input('Enter Desired File Name')
        ReIndexDF.to_pickle(str(New_Path) + fileName + ".pkl")
    print(ReIndexDF)
    #endregion
    return DF
# TimeFiller(MLsentDF, "T", True)
# TimeFiller(LXsentDF, "T", True)
# TimeFiller(ETHpriceDF, "T", True)


# LX_ETH_DF = pd.read_pickle(Pkl_Paths + "6DataSets/" + str( "LX_ETH.pkl"))
# ML_ETH_DF = pd.read_pickle(Pkl_Paths + "6DataSets/" + str( "ML_ETH.pkl"))


def NaNFiller(DF, method, save):
    #region ------ Imputing with method ----------
    New_Path = "Data/Pickled_Data/6DataSets/"
    if method == "mean":
        # - Imputing average to column of DF
        avg = np.mean(DF)
        DF = DF.fillna(avg)
    elif method == "interp":
        # - - Straight Interpolatin for each column of DF
        DF = DF.interpolate(method ='linear')
    elif method == "spline":
        # - - - Curved Interpolaiton for Cols of DF
        DF = DF.interpolate(method = 'spline',
                            order  =  4)
    elif method == "drop":
        # New_Path = "Data/Pickled_Data/Dropped_Data/"
        DF = DF.dropna( axis = 'rows' )
    else:
        print("Method spcecifed not recognised!")
    #endregion
    #region ----- Saving Data as Pickle  ---------
    if save == True:
        print("Enter Desired File Name:")
        fileName = input(New_Path + method + "_" )
        DF.to_pickle(str(New_Path) + method + fileName + ".pkl")
    print(DF)
    #endregion
    return DF

# NaNFiller(ML_ETH_DF, "interp", True)

# MultiPlot(NaNFiller(TmCs_PriceDF, "spline", True),"price_usd", 1, 6 , 12)

# MultiPlot(NaNFiller(ETHpriceDF, "drop", True),"price_usd", 1, 6 , 12)

# NaNFiller(MLsentDF, "drop", False)


def Data_Merger(DF1, DF2):
    #region ----- merging two dataframes ----------
    merged_DF = DF1.merge(DF2,
                          left_index = True,
                          right_index = True,
                          how = 'inner')
    #endregion
    #region ---- Saving DF as .pkl & .csv ---------
    # - Creating Path & FileName
    New_Path = "Data/Pickled_Data/6DataSets/"
    print("Enter Desired File Name:")
    fileName = input(New_Path)
    # - - Saving as .pkl & .csv
    merged_DF.to_pickle(New_Path + fileName + ".pkl")
    merged_DF.to_csv(   New_Path + fileName + ".csv")
    #endregion
    print(merged_DF)
    return merged_DF

# print(TmCs_PriceDF)
# print(TmCs_MLSntDF)
# print(TmCs_LXSntDF)



# Data_Merger(TmCs_PriceDF, TmCs_LXSntDF)


# Data_Merger(NaNFiller(ETHpriceDF, "drop", False), NaNFiller(MLsentDF, "drop", False))



# print(ETHpriceDF)
# print(MLsentDF)
# print(LXsentDF)


# print(LXsentDF)
# print(" - - - - - Dropped DF - - - - - - ")
# print(LXsentDF.dropna(axis = 'rows'))
