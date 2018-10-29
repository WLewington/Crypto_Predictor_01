#region =========   Dependancys & Data    ==========
import os
import warnings
import csv
# - - String Cleaning
import re 
import ast
# - Data Modeling
import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
# - - Machine Learing
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.models import load_model
# - - - Visualisation
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import figure
# import matplotlib.numerix as N
from DataPreProcessor import MultiPlot3

#endregion ====== Dependancys & Data  ==============
#region --------- Interface Cleaning ---------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings
import time
localTime = time.asctime(time.localtime(time.time()))
print(" -------------- DataManipulator - ENTRY AT " + str(localTime) + " ---------------- ")
#endregion --------- Interface Cleaning ---------------
# ========== System Peramiters & settings ==========
# "File Name"<str>, RowShift<int>, Plot<bool>, Data_to_use, train2test ratio 
Data_Paramiters = ['interpML_ETH_DF', 20, False, 10000, 0.8]
My_DPI = 400

# - - Pickled Data Paths In Heirachial Order
Pkl_Paths = "Data/Pickled_Data/"
TmCs_Path = "TimeContinuousData/"
No_NA_Path = "NoNA_Data/"



def Load_Models_Data(ModelName):
    #region --- Collecting Data Peramiters -------
    # - Opening Peramiters CSV as Pandas DF
    df = pd.read_csv("models/ModelPeramiters.csv")
    
    # - - Selecting Rows == ModelName, by Coloum "Peramiters"
    Data_df = df.loc[df["Model"] == ModelName].ix[:,"Peramiters"]
    # - - - Collecting Index
    idx = Data_df.index[0]
    #endregion
    #region ------ Converting Data format --------
    # - Getting Data Paramiters as string
    Data = Data_df[idx]
    # - - Convert str to list
    Data_Paramiters[:] = eval(Data)
    #endregion
    return Data_Paramiters


def load_data():
    #region ----- Importing & Organising Data ----
    fileName, TimeShift, Plot, Used_Data, train2test = Data_Paramiters

    # - Opening CSV into Panda DF from Directory
    R_Data_path = "Data/Pickled_Data/6DataSets/"
    Prc_DF = pd.read_csv(str(R_Data_path) + fileName + ".csv")
    X_Prc_DF, Y_Prc_DF = (Prc_DF, Prc_DF)

    # - - Removing TimeShift Rows from Tail & Head for X & Y
    Y_Prc_DF = Y_Prc_DF.shift(-TimeShift, axis='index')
    X_Prc_DF.drop(X_Prc_DF.tail(TimeShift).index, inplace = True)

    # - - - Removing NA Values
    X_Prc_DF = X_Prc_DF.dropna()
    Y_Prc_DF = Y_Prc_DF.dropna()

    # - - - - Selecting "start point of data" <int>
    Data_Start = len(X_Prc_DF) - Used_Data
    X_Prc_DF = X_Prc_DF[Data_Start:]
    Y_Prc_DF = Y_Prc_DF[Data_Start:]

    # - - - - - Selecting & Removing Columns From Y & X
    Y_Prc_DF = Y_Prc_DF.loc[:,"price_usd"]
    X_Prc_DF = X_Prc_DF.drop(["Date_and_Time"], axis = 1)
    #endregion
    #region ---------- Graphing Data -------------
    if Plot == True:
        plt.plot(Y_Prc_DF.loc[:len(X_Prc_DF)], 
                label = "Close Price")  
        plt.show()  
    #endregion
    #region ---------- Normalisation -------------
    # - Collecting Column names
    X_Cols = list(X_Prc_DF)
    # - - Normaliser label
    scale = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    # - - - Converts Panda DF to numpy array
    X_Prc_Ar = np.array(X_Prc_DF).reshape((len(X_Prc_DF), len(X_Cols)))
    Y_Prc_Ar = np.array(Y_Prc_DF).reshape((len(Y_Prc_DF), 1))
    # - - - - Normalise between -1 & 1
    X_Prc_Ar = scale.fit_transform(X_Prc_Ar)
    Y_Prc_Ar = scale.fit_transform(Y_Prc_Ar)
    #endregion
    #region ------ Test/Train Data split ---------
    # - Creating Train/Test data index
    train_end =  int(len(X_Prc_Ar) * train2test)
    # - - Splitting Training & Testing data 
    X_Train = X_Prc_Ar[:train_end,]
    X_Test  = X_Prc_Ar[train_end:,]
    Y_Train = Y_Prc_Ar[:train_end,]
    Y_Test  = Y_Prc_Ar[train_end:,]
    # - - - ReShaping into 1 dimensional Array
    X_Train = X_Train.reshape(X_Train. shape + (1,))
    X_Test  = X_Test.reshape (X_Test.  shape + (1,))
    #endregion
    #region ------- Original Pandas DF -----------
    # - Opening .pkl Pandas Data frame
    Original_DF = pd.read_pickle(R_Data_path + fileName + ".pkl")
    # - - Removing Training Rows 
    Original_DF = Original_DF[TimeShift:]
    Original_DF = Original_DF[Data_Start:]
    Original_DF = Original_DF[train_end:]
    #endregion 
    #region --------- Data Name Details ----------
    # - Number of rows used
    Rows = "_Rw-"  + str(Used_Data)
    # - - Number of Rows shifted between X & Y
    TS   = "_TS-" + str(TimeShift)
    # - - - Number Of Inputs
    XDim = "_XDim-"+ str(len(X_Cols))
    # - - - - Concatinating and Cleaning to make DataName
    DataName = fileName + XDim + Rows + TS
    DataName = DataName.replace(".","")
    #endregion
    return X_Train, X_Test, Y_Train, Y_Test, scale, DataName, Original_DF   
# load_data()


# (name with no.extension<string>)
def LSTM_Build(dimensions, DropOutRate, NumEpochs, batchSize):
    #region ---- Loading Conditioned Data --------
    X_Train, X_Test, Y_Train, Y_Test, scale, DataName, Original_DF = load_data()
    cols = np.shape(X_Train)[1]  # -- Number of Columns~Intput dimension
    #endregion
    #region ----- Building LSTM Network ----------
    # - Model Class, Linear Stack of layers
    model = Sequential()
    # - - Model Details(Batch_size, Input shape, )
    model.add(LSTM(dimensions,
                   input_shape      = (cols, 1),
                   activation       = 'tanh', 
                   inner_activation = 'hard_sigmoid') )
    # - - - Reducing Overfiting, shuts of nodes
    model.add(Dropout(DropOutRate))
    model.add(  Dense(output_dim      = 1,    # - single figure out put
                      activation      = 'linear'))
    # - - - - Defining, Loss Funciton, Optimizer and metrics
    model.compile(  loss= "mean_squared_error", 
                    optimizer = "adam")
    #endregion
    #region ----- Training LSTM Network ----------
    model.fit(X_Train, Y_Train, 
              batch_size = batchSize, 
              nb_epoch = NumEpochs, 
              shuffle= False)
    # - Displaying Model Spec's
    print(model.summary())
    # - - Displaying Error
    train_score = model.evaluate (X_Train, Y_Train, batch_size= 1)
    test_score  = model.evaluate (X_Test,  Y_Test,  batch_size= 1)
    print("MeanSqrError for Training Set: ", round( train_score, 4))
    print("MeanSqrError for Testing  Set: ", test_score)
    #endregion
    #region ------- Model Name Details -----------
    # - Number of rows used
    Nrns  = "_Nrns-"  + str(dimensions)
    # - - Number of Rows shifted between X & Y
    DrpRt = "_Drp-"  + str(DropOutRate)
    # - - - Train2Test Ratio 
    nEpc  = "Epoc-" + str(NumEpochs)
    # - - - - Train2Test Ratio 
    BSize = "_BSize-"+ str(batchSize)
    # ---- Concatinating & Cleaning to make Model Name
    ModelName = nEpc + BSize + Nrns + DrpRt
    ModelName = ModelName.replace(".","")
    ModelName = ModelName + "_" +  DataName
    #endregion
    #region ------ Saving LSTM Network -----------
    # - Creating ModelName from Model & Data Peramiters
    model.save('models/NewModels/' + ModelName + '.h5')
    # - - Appends Model name & DataParamiters
    with open('models/ModelPeramiters.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([ModelName,str(Data_Paramiters)])
    #endregion
    return model
# LSTM_Build(1000, 0.2, 20, 16)


# (name with no.extension<string>, Plot? <Bool>, SavePlot? <Bool>)
def Pred_with_Model(ModelName, Plot, SavePlot):
    #region ----- Loading Model & Data -----------
    # - loading saved LSTM model
    model = load_model("models/NewModels/" + str(ModelName) + ".h5") 
    # - - Calls funciton to change Data_Peramiters
    Load_Models_Data(ModelName)
    # - - - Loading data with relivant peramiters
    X_Train, X_Test, Y_Train, Y_Test, scale, DataName, Original_DF = load_data()
    cols = np.shape(X_Train)[1]  # -- Number of Columns~Intput dimension
    #endregion
    #region ------ Predicting with LSTM ----------
    # - Creating array of predicted value
    pred1 = model.predict(X_Test)
    # - - DeNormailising array to real figures
    pred1  = scale.inverse_transform(np.array(pred1 ).reshape((len(pred1 ), 1)))
    X_Test = scale.inverse_transform(np.array(X_Test).reshape((len(X_Test), cols)))
    Y_Test = scale.inverse_transform(np.array(Y_Test).reshape((len( Y_Test), 1)))
    #endregion
    #region ----- Plotting Model Results ---------
    if Plot == True:
        Line_Graph(True, ModelName, True, pred1, Y_Test)
    #endregion
    return pred1, Y_Test, Original_DF


def Line_Graph(Save, ModelName, CustomName, Y_axis1, Y_axis2):
    #region -------- Generating Title ------------
    if CustomName == True:
        print("*** Title already begins with 'Graph to show ' ***")
        title = input("Enter Desired Graph Name:")
    else:
        title = ModelName
        title = title.replace("_", " ")
        title = title.replace("-", ":")
    #endregion
    #region -------- Plot Dimensions -------------
    figure( num=None, 
            figsize=(8, 4), 
            facecolor='w', 
            edgecolor='k')
    #endregion
    #region ---------- Plot Data -----------------
    # -  Plots Predictions,
    plt.plot(Y_axis1, label = "Predictions")
    # - - Plots Original
    plt.plot(Y_axis2, label = "Actual")
    #endregion
    #region -------- Plot Asthetics --------------
    # - Legend; position, size, visuals
    plt.legend( loc='upper center', 
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True, 
                shadow=True, 
                ncol=2)
    # - - Graph Title
    plt.title("Graph to show {}".format(title))
    # - - - Puts "$" infront of Y axis
    tick = mtick.FormatStrFormatter('$%.0f')
    ax = plt.axes()
    ax.yaxis.set_major_formatter(tick)
    # - - - - Reduces White Boarder Size
    plt.tight_layout()
    #endregion
    #region --------- Saving Plot  ---------------
    if Save == True:
        plt.savefig("Graphs/Predictions/{}.png".format(ModelName), dpi = My_DPI)
    plt.show()
    #endregion


# ("MLmodelName"<string>, "LXmodelName"<string>, meanPerHour<int>)
def Data_4_Graph(MLModelName, LXModelName, AvgFreq):
    #region ----- Calling Pred_with_Model --------
    TempML = Pred_with_Model(MLModelName, False, False)
    TempLX = Pred_with_Model(LXModelName, False, False)
    #endregion
    #region ------ Assembling Data into DF --------
    TempDF = pd.DataFrame({"Actual": TempML[2]['price_usd'], 
                           "LXSent": TempLX[0][:,0], 
                           "MLSent": TempML[0][:,0]})
    #endregion
    #region ------- Calling MultiPlot3  -----------
    MultiPlot3(TempDF, AvgFreq)
    #endregion
    return (TempDF)

# Data_4_Graph("Epoc-20_BSize-16_Nrns-1000_Drp-02_dropML_ETH_DF_XDim-3_Rw-7200_TS-20",
#              "Epoc-20_BSize-16_Nrns-1000_Drp-02_dropLX_ETH_DF_XDim-3_Rw-7200_TS-20",
#              2)



# ("MLmodelName"<string>, "LXmodelName"<string>, meanPerHour<int>)
def RMSE_Func(MLModelName, LXModelName):
    #region ----- Calling Pred_with_Model --------
    TempML = Pred_with_Model(MLModelName, False, False)
    TempLX = Pred_with_Model(LXModelName, False, False)
    #endregion
    #region ------ Assembling Data into DF --------
    TempDF = pd.DataFrame({"Actual": TempML[2]['price_usd'], 
                           "LXSent": TempLX[0][:,0], 
                           "MLSent": TempML[0][:,0]})
    #endregion
    #region ------- Calling MultiPlot3  -----------
    print(TempDF)
    RMSE_1 = ((TempDF["MLSent"] - TempDF["Actual"]) ** 2).mean() ** .5
    RMSE_2 = ((TempDF["LXSent"] - TempDF["Actual"]) ** 2).mean() ** .5

    print(" ---------------- RMSE_Func OUTPUT -------------------")
    print("Machine Learnt: " + str(RMSE_1))
    print("Lexicon Method: " + str(RMSE_2))
    #endregion
    return (TempDF)

# TempDF = RMSE_Func("Epoc-20_BSize-16_Nrns-1000_Drp-02_interpML_ETH_DF_XDim-3_Rw-10000_TS-20",
#           "Epoc-20_BSize-16_Nrns-1000_Drp-02_interpLX_ETH_DF_XDim-3_Rw-10000_TS-20")


# ("MLmodelName"<string>, "LXmodelName"<string>, meanPerHour<int>)
def RMSE_Func2(MLDataNm, LXDataNm, DataFraq, AvgFreqList):
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
    merged_DF = merged_DF[["LXSent","MLSent","price_usd_x"]]
    # - - - - Renaming Collumns
    merged_DF.columns = ["LXSent","MLSent","price_usd"]

    # - - - - - Selecting Rows to Output
    Start_Point =  int(len(merged_DF) * DataFraq)
    merged_DF = merged_DF.ix[Start_Point:, merged_DF.columns]

    # - - - - - Normalising data frame
    merged_DF = (merged_DF-merged_DF.min())/(merged_DF.max()-merged_DF.min())
    #endregion
    #region ------ Creating rolling Avg collumn --------
    print('------------- Post average collumn --------------- ')
    for avgFreq in AvgFreqList:

        LX_DF = pd.DataFrame(data=merged_DF['LXSent'].rolling(avgFreq).mean())
        ML_DF = pd.DataFrame(data=merged_DF['MLSent'].rolling(avgFreq).mean())

        merged_DF['LXSent/{}'.format(avgFreq)] = merged_DF['LXSent'].rolling(avgFreq).mean()
        merged_DF['MLSent/{}'.format(avgFreq)] = merged_DF['MLSent'].rolling(avgFreq).mean()

        print(avgFreq)

    print(LX_DF)
    print(ML_DF)

    #endregion
    #region ------ Creating RMSE Value for DF ----------
    RMSECols = merged_DF.columns.tolist()
    RMSECols.remove("price_usd")

    SentRMSE = []

    for col in RMSECols:

        RMSE_Temp = ((merged_DF[col] - merged_DF["price_usd"]) ** 2).mean() ** .5       
        SentRMSE.append((RMSE_Temp,col))

    RMSE_DF = pd.DataFrame(SentRMSE)
    print(" ---------------- RMSE_Func OUTPUT -------------------")
    print(RMSE_DF)
    #endregion
    return (merged_DF)


# RMSE_Func2("dropML_ETH_DF","dropLX_ETH_DF", 0.8, [2,5,10,20,60])


# ("ModelName"<string>, "LXmodelName"<string>, meanPerHour<int>)
def Data_4_Graph2(ModelName, MLData ,AvgFreq):
    #region ----- Calling Pred_with_Model --------
    TempPred = Pred_with_Model(ModelName, False, False)
    #endregion
    #region ------ Assigning Name to Data -------
    if MLData == True:
        DataName = "MLSent"
    else:
        DataName = "LXSent"
    #endregion
    #region ------ Assembling Data into DF --------
    TempDF = pd.DataFrame({"Actual"   : TempPred[2]['price_usd'], 
                           DataName   : TempPred[0][:,0]})
    #endregion
    #region ------- Calling MultiPlot3  -----------
    MultiPlot3(TempDF, AvgFreq)
    #endregion
    return (TempDF)

# Data_4_Graph2("Epoc-20_BSize-16_Nrns-1000_Drp-02_dropLX_ETH_DF_XDim-3_Rw-7200_TS-20",
#                 False, 2)



def Two_Line_Graph(Save, 
                   ModelName, 
                   CustomName, 
                   Y_axis1, Y_axis2, Y_axis3):
    #region -------- Generating Title ------------
    if CustomName == True:
        print("*** Title already begins with 'Graph to show ' ***")
        title = input("Enter Desired Graph Name:")
    else:
        title = ModelName
        title = title.replace("_", " ")
        title = title.replace("-", ":")
    #endregion
    #region -------- Plot Dimensions -------------
    figure( num=None, 
            figsize=(8, 4), 
            facecolor='w', 
            edgecolor='k')
    #endregion
    #region ---------- Plot Data -----------------
    # -  Plots Predictions  1
    plt.plot(Y_axis1, label = "LX Predictions")
    # - - Plots Predictions 2
    plt.plot(Y_axis2, label = "ML Predictions")
    # - - Plots Original
    plt.plot(Y_axis3, label = "Actual")
    #endregion
    #region -------- Plot Asthetics --------------
    # - Legend; position, size, visuals
    plt.legend( loc='upper center', 
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True, 
                shadow=True, 
                ncol=2)
    # - - Graph Title
    plt.title("Graph to show {}".format(title))
    # - - - Puts "$" infront of Y axis
    tick = mtick.FormatStrFormatter('$%.0f')
    ax = plt.axes()
    ax.yaxis.set_major_formatter(tick)
    # - - - - Reduces White Boarder Size
    plt.tight_layout()
    #endregion
    #region --------- Saving Plot  ---------------
    if Save == True:
        plt.savefig("Graphs/Predictions/{}.png".format(ModelName), dpi = My_DPI)
    plt.show()
    #endregion

# Two_Line_Graph(True, "Epoc-20_BSize-16_Nrns-1000_Drp-02_dropLX_ETH_DF_XDim-3_Rw-7200_TS-20.h5", True, InterpLX, InterpML, InterpRaw)

# Pred_with_Model("Epoc-20_BSize-16_Nrns-1000_Drp-02_dropLX_ETH_DF_XDim-3_Rw-7200_TS-20", True, True)   

# Load_Models_Data("Epoc-1_BSize-16_Nrns-1000_Drp-02BTC_USD_XDim-5_Rw-1000_TS-100")


def SentTestFunc():
    #region --------- Importing Data ----------
    from TwitterScraper import MLSentAnalysis
    # - Loading Sample Tweet Data Frame
    df = pd.read_csv("Data/Raw_Data/ETHTweets.csv")
    # - - Renaming Lexicon Sentiment Column
    df.rename(columns = {"Polarity":"LXSent"}, inplace = True)

    Tweets = df["Tweet Text"].tolist()
    print(df)
    print(' -------------- Divide ------------- ')
    df["MLSent"] = MLSentAnalysis(Tweets)
    #endregion
    #region --------- Organising Data ----------
    # - - - Converting date string to Pandas Datetime obj
    df["Time and Date"] = pd.to_datetime(df["Time and Date"], 
                                            dayfirst = False, 
                                            yearfirst = True)
    # - - - - Setting Date as index and dropping its column
    df.index = df["Time and Date"]
    df = df.drop(["Time and Date"], axis = 1)
    #endregion
    #region ----- Saving Data as Pickle ----------
    New_Path = "Data/Raw_Data/SampleTweetData/"
    df.to_pickle(str(New_Path) + "ETHSentTweets" + ".pkl")
    #endregion
    #region ------- Saving Data as CSV -----------
    df.to_csv(str(New_Path)  + "ETHSentTweets" + ".csv")
    #endregion
    PostCount_DF = df['LXSent'].value_counts()
    print(PostCount_DF)
    print(PostCount_DF.describe())

# SentTestFunc()

# print(len(RawSentDF))

# print(RawSentDF.columns)

# print(RawSentDF['Tweet Text'].value_counts())

# PostCount_DF = RawSentDF['User Name'].value_counts()

# print(PostCount_DF)


