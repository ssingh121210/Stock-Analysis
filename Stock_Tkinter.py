import pandas as pd
from tkinter import *
import tkinter as tk
from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk



from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2TkAgg)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

from tkinter import messagebox

window = Tk()
window = ThemedTk(theme="radiance")
s = ttk.Style()
s.configure('my.TButton', font=('Helvetica', 30),background='sky blue')

def pl():
    import pandas as pd
    import numpy as np

    #to plot within notebook
    import matplotlib.pyplot as plt
    %matplotlib inline

    #setting figure size
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

    #for normalizing data
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
    if( var.get()=='APPLE'):
        df = pd.read_csv('AAPL_data.csv')
    elif (var.get()=='GOOGLE'):
        
        df = pd.read_csv('GOOGL_data.csv')
    elif(var.get()=='EBAY'):
        df = pd.read_csv('EBAY_data.csv')
    elif(var.get()=='AMAZON'):
        df = pd.read_csv('AMZN_data.csv')
    train = df[:987]
    valid = df[987:]
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']
    

#sorting
    data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]
    train = new_data[:987]
    valid = new_data[987:]

    x_train = train.drop('close', axis=1)
    y_train = train['close']
    x_valid = valid.drop('close', axis=1)
    y_valid = valid['close']
    import datetime as dt
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data['date']=new_data['date'].map(dt.datetime.toordinal)
    train = new_data[:987]
    valid = new_data[987:]
    x_train = train.drop('close', axis=1)
    y_train = train['close']
    x_valid = valid.drop('close', axis=1)
    y_valid = valid['close']
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg=PolynomialFeatures(degree=8)
    X_poly=poly_reg.fit_transform(x_train)
    len_reg2=LinearRegression()
    len_reg2.fit(X_poly,y_train)
    preds=len_reg2.predict(poly_reg.fit_transform(x_valid))
    plt.switch_backend('QT5Agg')
    valid['Predictions'] = 0
    valid['Predictions'] = preds

    valid.index = new_data[987:].index
    train.index = new_data[:987].index
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.xlabel("Time",fontsize=30)
    plt.ylabel("Closing Value",fontsize=30)
    plt.grid(color='grey')

    plt.plot(train['close'],label='ACTUAL CURVE',color='midnightblue')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.plot(valid[['close']],label='REAL VALUE',color='orange')
    plt.plot(valid['Predictions'],label='PREDICTION',color ='k')
    
def command():
    import pandas as pd
    import numpy as np

    #to plot within notebook
    import matplotlib.pyplot as plt
    %matplotlib inline

    #setting figure size
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

    #for normalizing data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
    if( var.get()=='APPLE'):
        df = pd.read_csv('AAPL_data.csv')
        
    elif (var.get()=='GOOGLE'):
        plt.title('GOOGLE')
        
        df = pd.read_csv('GOOGL_data.csv')
    elif(var.get()=='EBAY'):
        
        df = pd.read_csv('EBAY_data.csv')
    elif(var.get()=='AMAZON'):
        
        df = pd.read_csv('AMZN_data.csv')
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']
    plt.switch_backend('QT5Agg')
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10
    plt.title(var.get(), fontsize=30)
    df.index = df['date']
    plt.xlabel("Time",fontsize=30)
    plt.ylabel("Closing Value",fontsize=30)
    plt.grid(color='grey')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.plot(df['close'])
def Lr():
    import pandas as pd
    import numpy as np

    #to plot within notebook
    import matplotlib.pyplot as plt
    %matplotlib inline

    #setting figure size
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

    #for normalizing data
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
    if( var.get()=='APPLE'):
        df = pd.read_csv('AAPL_data.csv')
    elif (var.get()=='GOOGLE'):
        
        df = pd.read_csv('GOOGL_data.csv')
    elif(var.get()=='EBAY'):
        df = pd.read_csv('EBAY_data.csv')
    elif(var.get()=='AMAZON'):
        df = pd.read_csv('AMZN_data.csv')
    train = df[:987]
    valid = df[987:]
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']
    

#sorting
    data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]
    train = new_data[:987]
    valid = new_data[987:]

    x_train = train.drop('close', axis=1)
    y_train = train['close']
    x_valid = valid.drop('close', axis=1)
    y_valid = valid['close']
    import datetime as dt
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data['date']=new_data['date'].map(dt.datetime.toordinal)
    train = new_data[:987]
    valid = new_data[987:]
    x_train = train.drop('close', axis=1)
    y_train = train['close']
    x_valid = valid.drop('close', axis=1)
    y_valid = valid['close']

 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train,y_train)
    preds = model.predict(x_valid)
    plt.switch_backend('QT5Agg')
    valid['Predictions'] = 0
    valid['Predictions'] = preds

    valid.index = new_data[987:].index
    train.index = new_data[:987].index
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.xlabel("Time",fontsize=30)
    plt.ylabel("Closing Value",fontsize=30)
    plt.grid(color='grey')

    plt.plot(train['close'],label='ACTUAL CURVE',color='midnightblue')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.plot(valid[['close']],label='REAL VALUE',color='orange')
    plt.plot(valid['Predictions'],label='PREDICTION',color ='k')
def lstm():
    import pandas as pd
    import numpy as np

#to plot within notebook
    import matplotlib.pyplot as plt
    %matplotlib inline

#setting figure size
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

#for normalizing data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    if( var.get()=='APPLE'):
        df = pd.read_csv('AAPL_data.csv')
        from keras.models import load_model

        model = load_model('APPLE.h5')
   
    elif (var.get()=='GOOGLE'):
        
        df = pd.read_csv('GOOGL_data.csv')
        from keras.models import load_model

        model = load_model('GOOGLE.h5')
    elif(var.get()=='EBAY'):
        df = pd.read_csv('EBAY_data.csv')
        from keras.models import load_model

        model = load_model('EBAY.h5')
    elif(var.get()=='AMAZON'):
        df = pd.read_csv('AMZN_data.csv')
        from keras.models import load_model

        model = load_model('AMAZON.h5')
    train = df[:987]
    valid = df[987:]
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']

#sorting
    data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]
    #importing required libraries
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

#creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])
    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]

#setting index
    new_data.index = new_data.date
    new_data.drop('date', axis=1, inplace=True)

#creating train and test sets
    dataset = new_data.values

    train = dataset[0:987,:]
    valid = dataset[987:,:]

#converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)
    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    plt.switch_backend('QT5Agg')
    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = closing_price
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.xlabel("Time",fontsize=30)
    plt.ylabel("Closing Value",fontsize=30)
    plt.grid(color='grey')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.plot(train['close'],label='ACTUAL CURVE',color='midnightblue')
    plt.plot(valid[['close']],label='REAL VALUE',color='orange')
    plt.plot(valid['Predictions'],label='PREDICTION',color ='k')
def KNN():
    import pandas as pd
    import numpy as np

    #to plot within notebook
    import matplotlib.pyplot as plt
    %matplotlib inline

    #setting figure size
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

    #for normalizing data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
    if( var.get()=='APPLE'):
        df = pd.read_csv('AAPL_data.csv')
    elif (var.get()=='GOOGLE'):
        
        df = pd.read_csv('GOOGL_data.csv')
    elif(var.get()=='EBAY'):
        df = pd.read_csv('EBAY_data.csv')
    elif(var.get()=='AMAZON'):
        df = pd.read_csv('AMZN_data.csv')
    train = df[:987]
    valid = df[987:]
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']

#sorting
    data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]
    train = new_data[:987]
    valid = new_data[987:]

    x_train = train.drop('close', axis=1)
    y_train = train['close']
    x_valid = valid.drop('close', axis=1)
    y_valid = valid['close']
    import datetime as dt
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data['date']=new_data['date'].map(dt.datetime.toordinal)
    train = new_data[:987]
    valid = new_data[987:]
    x_train = train.drop('close', axis=1)
    y_train = train['close']
    x_valid = valid.drop('close', axis=1)
    y_valid = valid['close']
    from sklearn import neighbors
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(x_valid)
    x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
    model.fit(x_train,y_train)
    preds = model.predict(x_valid)
    plt.switch_backend('QT5Agg')
    valid['Predictions'] = 0
    valid['Predictions'] = preds
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.plot(train['close'],label='ACTUAL CURVE',color='midnightblue')
    plt.plot(valid[['close']],label='REAL VALUE',color='orange')
    plt.plot(valid['Predictions'],label='PREDICTION',color ='k')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.xlabel("Time",fontsize=30)
    plt.ylabel("Closing Value",fontsize=30)
    plt.grid(color='grey')
def MF():
    import pandas as pd
    import numpy as np

#to plot within notebook
    import matplotlib.pyplot as plt
    %matplotlib inline

#setting figure size
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

#for normalizing data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
    if( var.get()=='APPLE'):
        df = pd.read_csv('AAPL_data.csv')
    elif (var.get()=='GOOGLE'):
        
        df = pd.read_csv('GOOGL_data.csv')
    elif(var.get()=='EBAY'):
        df = pd.read_csv('EBAY_data.csv')
    elif(var.get()=='AMAZON'):
        df = pd.read_csv('AMZN_data.csv')
    train = df[:987]
    valid = df[987:]
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']
    
    train = df[:987]
    valid = df[987:]
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]
        df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']

    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']

#sorting
    data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]
    #split into train and validation
    train = new_data[:987]
    valid = new_data[987:]

    x_train = train.drop('close', axis=1)
    y_train = train['close']
    x_valid = valid.drop('close', axis=1)
    y_valid = valid['close']
    x_train

    preds = []
    for i in range(0,272):
        a = train['close'][len(train)-272+i:].sum() + sum(preds)
        b = a/272
        preds.append(b)

   


    plt.switch_backend('QT5Agg')
    valid['Predictions'] = 0
    valid['Predictions'] = preds
    plt.xlabel("Time")
    plt.ylabel("Close")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.plot(train['close'],label="TIME")
    plt.plot(valid[['close', 'Predictions']])
    
    

   
    
    
    
 
    
    
w, h = window.winfo_screenwidth(), window.winfo_screenheight()
from tkinter import Menu
menu = Menu(window)
 
new_item = Menu(menu)
 
new_item.add_command(label='New')
new_item.add_separator()
 
new_item.add_command(label='Edit')
new_item.add_separator()
new_item.add_command(label='Exit')
 
menu.add_cascade(label='File', menu=new_item)
window.configure()
window.config(menu=menu)
 
window.title("Stock Analysis  app")
 
window.geometry("%dx%d+0+0" % (w, h))
 
# lbl = Label(window, text="Hello")
 
# lbl.grid(column=0, row=0)

var = StringVar(window)




l1 = ttk.Label(window,text="Pick An Algorithm",font=("Tahoma", 25, 'bold'),background='light sky blue')
l2 = ttk.Label(window,text="Pick A Company",font=("Tahoma", 12, 'bold'),background='light sky blue')

# homescreenImage = tk.PhotoImage(file="SM.png") 

# homescreenFrame.grid()
# homescreenLabel = tk.Label(window, image=homescreenImage)
# homescreenLabel.grid(column=0,row=2)


 
btn = ttk.Button(window, text="Select")
btn1 = ttk.Button(window, text ="Graph",width=100,command=command,style='my.TButton')
btn2 = ttk.Button(window, text ="Mathematical Function",width=100,command=MF)
btn3 = ttk.Button(window, text ="Linear Regrssion Algorithm",width=100,command=Lr)
btn4 = ttk.Button(window, text ="Polynomial Regrssion Algorithm",width=100,command=pl)
btn5 = ttk.Button(window, text ="Nearest neighbour Algorithm",width=100,command=KNN)
btn6 = ttk.Button(window, text ="Long Short-Term Memory Algorithm",width=100,command=lstm)
# btn1.grid(padx=50,pady=10)
# btn2.grid(padx=50,pady=10)
# btn3.grid(padx=50,pady=10)
# btn4.grid(padx=50,pady=10)
# btn5.grid(padx=50,pady=10)
# btn6.grid(padx=50,pady=10)

l2.grid(padx=30, pady=30,column= 0 ,row=0)
l1.grid(padx=30, pady=30,column= 13 ,row=1)
btn.grid(padx=10, pady=10,column=2, row=0)
btn1.grid(padx=30, pady=20,column=13, row=2)
btn2.grid(padx=30, pady=20,column=13, row=3)
btn3.grid(padx=30, pady=20,column=13, row=4)
btn4.grid(padx=30, pady=20,column=13, row=5)
btn5.grid(padx=30, pady=20,column=13, row=6)
btn6.grid(padx=30, pady=20,column=13, row=7)

option =ttk.OptionMenu(window, var, 'SELECT','APPLE', 'GOOGLE', 'EBAY','AMAZON','TATA INC') 
# option.grid(column=0, row=0)
option.grid(column=1,row=0)



window.configure(background = 'sky blue')
 
window.mainloop()
