# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfile 
import random
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
def plotData(data):
    fig.clear()
    plot = fig.add_subplot(111)
    plot.bar(data['prognosis'].value_counts().keys(),data['prognosis'].value_counts(normalize = True))
    canvas.draw()
def utilizeModel():
    file_path= askopenfile(mode='r', filetypes=[('Csv Files', '*csv')])
    unknownData= pd.read_csv(file_path)
    prediction=model.predict(unknownData)
    dataf=pd.DataFrame(prediction,columns=['prognosis'],index=None)
    dataf.to_csv('prediction.csv')
    plotData(dataf)
def readData():
   global data
   file_path= askopenfile(mode='r', filetypes=[('Csv Files', '*csv')])
   if file_path is None:
       pass
   else:
       tk.Label(text="Current dataset: \n"+file_path.name.split('/')[-1]).place(x=20,y=160)
       tk.Button(window,text="train model",command=startTraining).place(x=25,y=200)
       data= pd.read_csv(file_path)
       plotData(data)
def displayAccuracy(accuracy):
    print(type(accuracy))
    if (type(accuracy) is list):
        accuracy_label["text"]="model accuracy: \n{:.2f} {:.2f} {:.2f} {:.2f} %".format(*accuracy)
        return
    accuracy_label["text"]="model accuracy: {:.2f}%".format(accuracy*100)
def startTraining():
    model_name=model_selection.get()
    if(len(model_name)==0):
        messagebox.showinfo("showinfo","Please Select a training model")
        return
    dataset = data.drop("prognosis",axis=1)
    prognosis = data["prognosis"]
    plt.subplots_adjust(left = 0.9, right = 2 , top = 2, bottom = 1)
    x_train,x_test,y_train,y_test = train_test_split(dataset,prognosis,test_size=0.3,random_state=42)
    try:
        if(hyperparamaeters_entry['state']=='normal'):
            hp = int(hyperparamaeters_entry.get())
    except ValueError :
        messagebox.showinfo("showinfo","Please enter a valid hyperparameter")
        return
    global model
    result = None
    if(model_name=='Random Froest'):
        model= RandomForestClassifier(max_depth=hp)
    elif(model_name=='Decision Tree'):
        model= DecisionTreeClassifier(max_depth=hp)
    elif(model_name=='random search cross validation'):
        result= performRS(x_train, y_train)
    elif(model_name=='KNNeighbour'):
        model = KNeighborsClassifier(hp)
    elif(model_name=='Navie bayes'):
        model= GaussianNB()
    if (model!=None):
        model.fit(x_train,y_train)
        displayAccuracy(model.score(x_test,y_test))
    else:
        displayAccuracy([m.best_estimator_.score(x_test,y_test)*100 for m in result])
        model = result[(random.randint(0,len(result)))].best_estimator_
    tk.Button(window,text="utilize the model",command=utilizeModel).place(x=12,y=400)
def onModelSelection(model_selection):
    model_name= model_selection.widget.get()
    labels= {"KNNeighbour":"Enter the value of K hyperparameter","Decision Tree":"Enter the value of the depth hyperparameter","Random Froest":"Enter the value of the depth hyperparameter"}
    if (model_name in labels ):
        hyperparamaeters_label.config(text=labels[model_name])
        hyperparamaeters_entry.config(state='normal')
    else:
        hyperparamaeters_label.config(text="This model has no available hyperparamter")
        hyperparamaeters_entry.config(state='disabled')
def performRS(x,y):
    tree_components= {
        "criterion":["gini","entropy"],
      "max_depth":list(range(3,data.shape[1]))
      }
    knn__n_neighbors= {
        "n_neighbors": list(range(1, 31))
        }
    models= [(RandomForestClassifier(),tree_components), #random Forest is a set of various decision trees so we use the same list of hyperparameters
     (DecisionTreeClassifier(),tree_components), 
     (KNeighborsClassifier(),knn__n_neighbors),
     (GaussianNB(),{}) #navie bayes model doesn't require any  hyperparamaeters
     ]
    results=[]
    for model in models:
        clf = RandomizedSearchCV(model[0],model[1],return_train_score=True, cv=4)
        clf.fit(x,y)
        test_scores = clf.cv_results_['mean_test_score']
        train_scores = clf.cv_results_['mean_train_score'] 
        plt.plot(test_scores, label='test')
        plt.plot(train_scores, label='train')
        plt.legend(loc='best')
        plt.show()
        results.append(clf)
    return results
data= None
window = tk.Tk()
model= None
window.title("Clinical decision support system")
hyperparamaeters_entry = tk.Entry( window,width=10,state='disabled' )
hyperparamaeters_entry.place(x=25,y=285)
model_selection = ttk.Combobox(window,state="readonly",width=10,textvariable=tk.StringVar())
model_selection['values']= ('KNNeighbour','Decision Tree','Random Froest', 'Navie bayes',"random search cross validation")
upld = tk.Button(window, text='Upload dataset', command=readData)
hyperparamaeters_label = tk.Label(text="hyperparameters configuration",wraplength=100)
hyperparamaeters_label.place(x=10,y=230)
tk.Label(text="Project 4 Clinical decision support system").place(x=150,y=0) #project title
tk.Label(text="Model type").place(x=20,y=60) #Label of combobox
model_selection.place(x=20,y=90)
fig = Figure(figsize=(4,4))
canvas = FigureCanvasTkAgg(fig, master=window)
toolbar = NavigationToolbar2Tk(canvas, window)
toolbar.update()
tk.Label(text="Visualization of the dataset").place(x=230,y=400) #Label of combobox
canvas.draw()
canva_tk= canvas.get_tk_widget()
canva_tk.place(x=160,y=90)
upld.place(x=20,y=120)
accuracy_label = tk.Label(text="model accuracy: ") #Label of accuracy
accuracy_label.place(x=12,y=350)
model_selection.current()
model_selection.bind("<<ComboboxSelected>>", onModelSelection)
screen_width= window.winfo_screenwidth()
screen_height=window.winfo_screenheight()
window.geometry("500x500+"+str(int(screen_width/2-(500/2)))+"+"+str(int(screen_height/2-(400/1.7))))
window.iconbitmap("./icon.ico")
window.mainloop()
