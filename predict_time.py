# Importing libraries
import tkinter as tk
import pandas as pd
import numpy as np
import math
import operator
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#list for loading the data
X = [] 
y = []  
HEIGHT = 700
WIDTH = 1000

#Preprocessing data 
data = pd.read_csv("TRAIN_SET.csv")
data.dropna(inplace=True)
print(data.columns)
df=data.to_csv('train_set_cleaned.csv')
df = pd.read_csv("train_set_cleaned.csv")

#Importing data
print("*************DataSet**************")
print(df.head())
X = df[['Zone', 'CodedDay', 'CodedWeather', 'Temperature']].copy()
y = df['Realtime'].copy()
print(X)
print(y)

#KNN accuracy
neigh = KNeighborsRegressor()
y_pred=neigh.fit(X, y) 
h=neigh.score(X,y)
print()
print("Accuracy of KNN is ",h," that is ",math.ceil(h*100),"%")

#LinearRegression accuracy
model = LinearRegression()
model.fit(X, y)
t=model.score(X, y)
print()
print("Accuracy of Linear Regression is: ",t," that is ",math.ceil(t*100),"%")

#SVR accuracy
logreg = SVR(gamma='scale')
logreg.fit(X, y)
l=logreg.score(X,y)
print()
print("Accuracy of SVR is: ",l," that is ",math.ceil(l*100),"%")

#RandomForestRegressor accuracy
regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
regr.fit(X, y)
rand=regr.score(X,y)
print()
print("Accuracy of Random Forest is: ",rand," that is ",math.ceil(rand*100),"%")
print()

#Function to check whether the output Final string is valid
#check for any exceptions 
def format_response(time):
    try:

        final_str = 'Time Taken: %s minutes' % (time)
    except:
        final_str = 'There was a problem retrieving that information'

    return final_str


# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


# Defining our KNN model
def knn(trainingSet, testInstance, k):
 
    distances = {}
    sort = {}
 
    length = testInstance.shape[1]
    
    #### 
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        
        #### 
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]

        ####
 
    #### 
    # Sorting them on the basis of distance

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    #### 
 
    neighbors = []
    
    #### 
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    ####
    #print(neighbors)
    classVotes = {}
    
    ####
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    ####

    #### 
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    label['text'] = format_response(str(sortedVotes[0][0]))
    #return(sortedVotes[0][0], neighbors)
    #### 

print("**********************************INSTRUCTIONS**********************************")
print()
print("Time of the day (Zone column) : ")
print("A number code representing a 10 minute interval timezone, splitting the 24 hours of a day into 144 zones,")
print("( For example, the 10 minute duration from 00:00 to 00:10 Hrs is coded as 1 and 00:10 to 00:20 Hrs is coded as 2, and so on) ")
print("--------------------------------------------------------------------------------")
print("Day of the week (CodedDay column) :")
print("Week day in a coded number, 7 weekdays converted into to 7 numbers starting from 1(Sunday) to 7(Saturday).")
print("--------------------------------------------------------------------------------")
print("CodeWeather :")
print("11 to 30 (Cold to High Temperature)")
print("--------------------------------------------------------------------------------")
print("Temperature (Temperature column) :")
print("Average temperature during the day, in Fahrenheit. ")
print()   
root = tk.Tk()
# Code to add widgets...

#Creating a rectangular area
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

#attaching a background image
background_image = tk.PhotoImage(file='landscape.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

#creating a frame for labels,entrys and button
frame = tk.Frame(root, bg='#80c1ff', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=1, relheight=0.3, anchor='n')


#creating 4 labels
label1 = tk.Label(frame,text="Zone")
label1.grid(row=0,column=0)

label2 = tk.Label(frame,text="CodeDay")
label2.grid(row=0,column=1)

label3 = tk.Label(frame,text="CodeWeather")
label3.grid(row=0,column=2)

label4 = tk.Label(frame,text="Temperature")
label4.grid(row=0,column=3)


#creating 4 entry widgets
entry = tk.Entry(frame, font=40)
entry.grid(row=1,column=0)

entry2 = tk.Entry(frame, font=40)
entry2.grid(row=1,column=1)

entry3 = tk.Entry(frame, font=40)
entry3.grid(row=1,column=2)

entry4 = tk.Entry(frame, font=40)
entry4.grid(row=1,column=3)

#print(type(entry.get()))

#declaring our parameter as 5
k=5


button = tk.Button(frame, text="Predict Time", font=40, command=lambda: knn(data, pd.DataFrame([[int(entry.get()),int(entry2.get()),int(entry3.get()),int(entry4.get())]]), k))
button.grid(row=1,column=4)


lower_frame = tk.Frame(root, bg='#800080', bd=10)
lower_frame.place(relx=0.5, rely=0.25, relwidth=0.75, relheight=0.6, anchor='n')

#label for printing the predicted time
label = tk.Label(lower_frame)
label.place(relwidth=1, relheight=1)

#END
root.mainloop()
