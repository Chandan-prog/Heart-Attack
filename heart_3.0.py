
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np

dataset = pd.read_csv('C:\\Users\\mishr\\Desktop\\heart.csv')
x = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:,[13]].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

def knn():
    global acc_knn
    from sklearn.neighbors import KNeighborsClassifier
    K = KNeighborsClassifier(n_neighbors=5)

    K.fit(x_train,y_train)
    
    y_pred_knn = K.predict(x_test)
    
    from sklearn.metrics import accuracy_score
    acc_knn = accuracy_score(y_test,y_pred_knn)
    acc_knn = round(acc_knn*100,2)
    m.showinfo(title="KNN",message="accuracy is"+str(acc_knn)+"%")
    

def dt():
    global acc_dt
    from sklearn.tree import DecisionTreeClassifier
    D = DecisionTreeClassifier()
    D.fit(x_train,y_train)
    Y_pred_dt = D.predict(x_test)
    
    from sklearn.metrics import accuracy_score
    acc_dt = accuracy_score(y_test,Y_pred_dt)
    acc_dt = round(acc_dt*100,2)
    m.showinfo(title="Decision Tree",message="accuracy is"+str(acc_dt)+"%")
    
def nb():
    
    global acc_nb
    from sklearn.naive_bayes import GaussianNB
    N = GaussianNB()
    N.fit(x_train,y_train)
    Y_pred_nb = N.predict(x_test)
    
    from sklearn.metrics import accuracy_score
    acc_nb = accuracy_score(y_test,Y_pred_nb)
    acc_nb = round(acc_nb*100,2)
    m.showinfo(title="naive bayes",message="accuracy is"+str(acc_nb)+"%")

def rf():
    
    global acc_rf
    from sklearn.ensemble import RandomForestClassifier
    Rclassifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    Rclassifier.fit(x_train,y_train)
    y_pred_rf = Rclassifier.predict(x_test)
    
    from sklearn.metrics import accuracy_score
    acc_rf = accuracy_score(y_test,y_pred_rf)
    acc_rf = round(acc_rf*100,2)
    m.showinfo(title="random forest",message="accuracy is"+str(acc_rf)+"%")
    
def svm():
    global acc_svm
    global Sclassifier
    from sklearn.svm import SVC
    Sclassifier = SVC(kernel = 'rbf', random_state = 0)
    Sclassifier.fit(x_train,y_train)
    y_pred_svm = Sclassifier.predict(x_test)
    
    from sklearn.metrics import accuracy_score
    acc_svm = accuracy_score(y_test,y_pred_svm)
    acc_svm = round(acc_svm*100,2)
    m.showinfo(title="Support Vactor Machine",message="accuracy is"+str(acc_svm)+"%")
    
def compare():
    model=["KNN","NB","DT","RF","SVM"]
    accuracy=[acc_knn,acc_nb,acc_dt,acc_rf,acc_svm ]
    plt.bar(model,accuracy,color=["orange","blue","green","yellow","red"])
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()
    
def submit():
    age = int(v1.get())
    sex = int(v2.get())
    cp = int(v3.get())
    trestbps = int(v4.get())
    chol = int(v5.get())
    fbs = int(v6.get())
    restecg = int(v7.get())
    thalach = int(v8.get())
    exang = int(v9.get())
    oldpeak = float(v10.get())
    slope = int(v11.get())
    ca= int(v12.get())
    thal = int(v13.get())
    a = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    a = sc.transform(a)
    result = Sclassifier.predict(a)
    
    if result==0:
        RES="Congratulations! Your report is negative"
    elif result==1:
        RES="Sorry! Your report is positive"
    m.showinfo(title="HEART DISEASE",message=RES)
    
def reset():
    v1.set("")
    v2.set("")
    v3.set("")
    v4.set("")
    v5.set("")
    v6.set("")
    v7.set("")
    v8.set("")
    v9.set("")
    v10.set("")
    v11.set("")
    v12.set("")
    v13.set("")

from tkinter import *
import tkinter.messagebox as m

w=Tk()

v1 = StringVar()
v2 = StringVar()
v3 = StringVar()
v4 = StringVar()
v5 = StringVar()
v6 = StringVar()
v7 = StringVar()
v8 = StringVar()
v9 = StringVar()
v10 = StringVar()
v11 = StringVar()
v12 = StringVar()
v13 = StringVar()

B1 = Button(w,text="KNN",command=knn,font=("arial",20 ,"bold"))
B2 = Button(w,text="NB",command=nb,font=("arial",20 ,"bold"))
B3 = Button(w,text="DT",command=dt,font=("arial",20 ,"bold"))
B4 = Button(w,text="RF",command=rf,font=("arial",20 ,"bold"))
B5 = Button(w,text="SVM",command=svm,font=("arial",20 ,"bold"))
B6 = Button(w,text="Compare",command=compare,font=("arial",20,"bold"))
B7 = Button(w,text="submit",command=submit,font=("arial",20,"bold"))
B8 = Button(w,text="reset",command=reset,font=("arial",20,"bold"))

L1 = Label(w,text="Age",font=("arial",20,"bold"))
L2 = Label(w,text="Sex",font=("arial",20,"bold"))
L3 = Label(w,text="Chest Pain Type",font=("arial",20,"bold"))
L4 = Label(w,text="Resting Blood Pressure ",font=("arial",20,"bold"))
L5 = Label(w,text="Serum Cholestoral(mg/dl)",font=("arial",20,"bold"))
L6 = Label(w,text="Fasting Blood Sugar(>120)",font=("arial",20,"bold"))
L7 = Label(w,text="Resting ECG Results(0,1,2)",font=("arial",20,"bold"))
L8 = Label(w,text="MAX Heart Rate",font=("arial",20,"bold"))
L9 = Label(w,text="Exercise Induced Anigma",font=("arial",20,"bold"))
L10 = Label(w,text="Old Peak(ST Depression)",font=("arial",20,"bold"))
L11 = Label(w,text="Slope(Peak Exercise ST segment)",font=("arial",20,"bold"))
L12 = Label(w,text="Number of major vessels(0-3)",font=("arial",20,"bold"))
L13 = Label(w,text="thal(3,6,7)",font=("arial",20,"bold"))

E1 = Entry(w,textvariable=v1,font=("arial",20,"bold"))
E2 = Entry(w,textvariable=v2,font=("arial",20,"bold"))
E3 = Entry(w,textvariable=v3,font=("arial",20,"bold"))
E4 = Entry(w,textvariable=v4,font=("arial",20,"bold"))
E5 = Entry(w,textvariable=v5,font=("arial",20,"bold"))
E6 = Entry(w,textvariable=v6,font=("arial",20,"bold"))
E7 = Entry(w,textvariable=v7,font=("arial",20,"bold"))
E8 = Entry(w,textvariable=v8,font=("arial",20,"bold"))
E9 = Entry(w,textvariable=v9,font=("arial",20,"bold"))
E10 =Entry(w,textvariable=v10,font=("arial",20,"bold"))
E11 =Entry(w,textvariable=v11,font=("arial",20,"bold"))
E12 =Entry(w,textvariable=v12,font=("arial",20,"bold"))
E13 =Entry(w,textvariable=v13,font=("arial",20,"bold"))


B1.grid(row=1,column=1)
B2.grid(row=2,column=1)
B3.grid(row=3,column=1)
B4.grid(row=4,column=1)
B5.grid(row=5,column=1)
B6.grid(row=6, column=1)
B7.grid(row=7, column=1)
B8.grid(row=8, column=1)

L1.grid(row=1,column=2)
L2.grid(row=2,column=2)
L3.grid(row=3,column=2)
L4.grid(row=4,column=2)
L5.grid(row=5,column=2)
L6.grid(row=6,column=2)
L7.grid(row=7,column=2)
L8.grid(row=8,column=2)
L9.grid(row=9,column=2)
L10.grid(row=10,column=2)
L11.grid(row=11,column=2)
L12.grid(row=12,column=2)
L13.grid(row=13,column=2)

E1.grid(row=1,column=3)
E2.grid(row=2,column=3)
E3.grid(row=3,column=3)
E4.grid(row=4,column=3)
E5.grid(row=5,column=3)
E6.grid(row=6,column=3)
E7.grid(row=7,column=3)
E8.grid(row=8,column=3)
E9.grid(row=9,column=3)
E10.grid(row=10,column=3)
E11.grid(row=11,column=3)
E12.grid(row=12,column=3)
E13.grid(row=13,column=3)
w.mainloop()

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
confusn_mat = confusion_matrix(y_test, y_pred)