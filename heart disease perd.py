#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("C:/Users/Kartik yadav/OneDrive/Desktop/heart.csv")


# In[ ]:


data.head()#display top 5 rows of the dataset


# In[ ]:


data.tail()#display last 5 rows of the dataset


# In[ ]:


data.shape


# In[ ]:


print("Number of rows",data.shape[0])
print("Number of columns",data.shape[1])


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data_dup = data.duplicated().any()
print(data_dup)


# In[ ]:


data = data.drop_duplicates()


# In[ ]:


data.shape


# In[ ]:


data_dup= data.duplicated().any


# In[ ]:


data_dup


# In[ ]:


data.describe()


# In[ ]:


#correlation matrix
plt.figure(figsize=(17,6))
sns.heatmap(data.corr(),annot=True)


# In[ ]:


data.columns


# In[ ]:


data['target'].value_counts()


# In[ ]:


sns.countplot(data['target'])


# In[ ]:


data.columns


# In[ ]:


data['sex'].value_counts()


# In[ ]:


sns.countplot(data['sex'])
plt.xticks([0,1],['Female','Male'])


# In[ ]:


sns.countplot(x='sex',hue='target',data=data)
plt.xticks([1,0],['Male','Female'])
plt.legend(labels=['No-Disease','Disease'])
plt.show


# In[ ]:


sns.distplot(data['age'],bins=20)
plt.show()


# In[ ]:


sns.countplot(data['cp'])
plt.xticks([0,1,2,3],{"typical angina","atypical angina","non-anginal pain","asymptomatic"})
plt.xticks(rotation=75)
plt.show()


# In[ ]:


data.columns


# In[ ]:


sns.countplot(x='cp',hue='target',data=data)
plt.legend(labels=["No-Disease","Disease"])
plt.show()


# In[ ]:


sns.countplot(x='fbs',hue='target',data=data)
plt.legend(labels=["No-Disease","Disease"])
plt.show()


# In[ ]:


data.columns


# In[ ]:


data['trestbps'].hist()


# In[ ]:


g= sns.FacetGrid(data,hue="sex",aspect=4)
g.map(sns.kdeplot,'trestbps',shade=True)
plt.legend(labels=['Male','Female'])


# In[ ]:


data.columns


# In[ ]:


data['chol'].hist()


# In[ ]:


data.columns


# In[ ]:


#data Processing
cate_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique()<=10:
        cate_val.append(column)
    else:
        cont_val.append(column)


# In[ ]:


cate_val


# In[ ]:


cont_val


# In[ ]:


data.hist(cont_val,figsize=(15,6))
plt.tight_layout()
plt.show


# In[ ]:


#encoding categorical data
cate_val


# In[ ]:


data['cp'].unique()


# In[ ]:


cate_val.remove('sex')
cate_val.remove('target')
pd.get_dummies(data, columns= cate_val,drop_first=True)


# In[ ]:


#featuring Scaling
data.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:



st=StandardScaler()
data[cont_val]= st.fit_transform(data[cont_val])


# In[ ]:


data.head()


# In[ ]:


#splitting the dataset into the training set and test set
x=data.drop('target',axis=1)


# In[ ]:


y=data['target']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


x_train


# In[ ]:


x_test


# In[ ]:


#logistic regression
data.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


log = LogisticRegression()
log.fit(x_train,y_train)


# In[ ]:


y_pred1=log.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


a1=accuracy_score(y_test,y_pred1)


# In[ ]:


a1


# In[ ]:


#SVC
from sklearn import svm


# In[ ]:


svm = svm.SVC()


# In[ ]:


svm.fit(x_train,y_train)


# In[ ]:


y_pred2 = svm.predict(x_test)


# In[ ]:


a2=accuracy_score(y_test,y_pred2)


# In[ ]:


a2


# In[ ]:


#kneighbours classifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=  KNeighborsClassifier()


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


y_pred3 = knn.predict(x_test)


# In[ ]:


a3=accuracy_score(y_test,y_pred3)


# In[ ]:


a3


# In[ ]:


score= []

for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    score.append(accuracy_score(y_test,y_pred))


# In[ ]:


score


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
accuracy_score(y_test,y_pred)


# In[ ]:


#non linerar ml algorithm
#desicion tree classifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


data = pd.read_csv("C:/Users/Kartik yadav/OneDrive/Desktop/heart.csv")


# In[ ]:


data=data.drop_duplicates()


# In[ ]:


data.shape


# In[ ]:


x= data.drop('target', axis=1)
y=data['target']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


dt=DecisionTreeClassifier()


# In[ ]:


dt.fit(x_train,y_train)


# In[ ]:


y_pred4=dt.predict(x_test)


# In[ ]:


a4=accuracy_score(y_test,y_pred4)


# In[ ]:


a4


# In[ ]:


#Random forest Classifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier()


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


y_pred5=rf.predict(x_test)


# In[ ]:


a5=accuracy_score(y_test,y_pred5)


# In[ ]:


a5


# In[ ]:


#gradient Boosting Classifier


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gbc = GradientBoostingClassifier()


# In[ ]:


gbc.fit(x_train,y_train)


# In[ ]:


y_pred6=gbc.predict(x_test)


# In[ ]:


a6=accuracy_score(y_test,y_pred6)


# In[ ]:


a6


# In[ ]:


#compare between different models
final_data = pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                          'Acc':[a1,a2,a3,a4,a5,a6]})


# In[ ]:


final_data


# In[ ]:


#visualization of model accuracy
import seaborn as sns


# In[ ]:


sns.barplot(final_data['Models'],final_data['Acc'])


# In[ ]:


x= data.drop('target', axis=1)
y=data['target']


# In[ ]:


x.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier()
rf.fit(x,y)


# In[ ]:


#prediction on new data
import pandas as pd


# In[ ]:


new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slope':2,
    'ca':2,
    'thal':3,
},index=[0])


# In[ ]:


new_data


# In[ ]:


p=rf.predict(new_data)
if p[0]==0:
    print("NO Disease")
else:
    print('Disease')


# In[ ]:


#save model using joblib
import joblib


# In[ ]:


joblib.dump(rf,'Model_heart')


# In[ ]:


model=joblib.load('Model_heart')


# In[ ]:


model.predict(new_data)


# In[ ]:


#gui
from tkinter import *
import joblib
from tkinter import Button
from tkinter import PhotoImage


# In[ ]:



root =Tk()

root.title("Heart Diseases Prediction System")

root.geometry('600x1200')

root.config(bg="lightblue")

header=Label(root, text="Heart Diseases Prediction System",bg="lightblue"
      ,foreground="black",font=("Arial",15,"bold"))
header.pack()

try:

    image_path=PhotoImage(file=r"C:\Users\Kartik yadav\Downloads\heart.png")
    bg_image=Label(root,image=image_path)
    bg_image.place(relheight=1,relwidth=1)
    
    



    frame1= Frame(root, bg="lightblue")
    frame1.pack()
    l0=Label(frame1,text="Heart diseases Prediction",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l0.grid(row=0,column=0,pady=10)
#1
    l1=Label(frame1,text="Enter your Age",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l1.grid(row=1,column=0,pady=10)

    e1=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e1.grid(row=1,column=1,pady=10)
#2
    l2=Label(frame1,text="Enter Male or female as(1/0)",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l2.grid(row=2,column=0,pady=10)

    e2=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e2.grid(row=2,column=1,pady=10)
#3
    l3=Label(frame1,text="Enter value of chest pain",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l3.grid(row=3,column=0,pady=10)

    e3=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e3.grid(row=3,column=1,pady=10)
#4
    l4=Label(frame1,text="Enter value of resting blood pressure",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l4.grid(row=4,column=0,pady=10)

    e4=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e4.grid(row=4,column=1,pady=10)
#5
    l5=Label(frame1,text="Enter value of serum cholestrol",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l5.grid(row=5,column=0,pady=10)
    
    e5=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e5.grid(row=5,column=1,pady=10)
#6
    l6=Label(frame1,text="Enter value of fasting blood sugar",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l6.grid(row=6,column=0,pady=10)
    
    e6=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e6.grid(row=6,column=1,pady=10)
#7
    l7=Label(frame1,text="Enter value of resting electrocardiograpical results",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l7.grid(row=7,column=0,pady=10)
    
    e7=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e7.grid(row=7,column=1,pady=10)
#8
    l8=Label(frame1,text="Enter value of maximum heart rate achieved",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l8.grid(row=8,column=0,pady=10)
    
    e8=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e8.grid(row=8,column=1,pady=10)
#9
    l9=Label(frame1,text="Enter value of exercise induced angina",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l9.grid(row=9,column=0,pady=10)
    
    e9=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e9.grid(row=9,column=1,pady=10)
#10
    l10=Label(frame1,text="Enter value of oldpeak",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l10.grid(row=10,column=0,pady=10)
    
    e10=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e10.grid(row=10,column=1,pady=10)
#11
    l11=Label(frame1,text="Enter value of slope",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l11.grid(row=11,column=0,pady=10)

    e11=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e11.grid(row=11,column=1,pady=10)
#12
    l12=Label(frame1,text="Enter value of number of major vessels(0-3)",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l12.grid(row=12,column=0,pady=10)

    e12=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e12.grid(row=12,column=1,pady=10)
#13
    l13=Label(frame1,text="Enter value of thal",bg="lightblue",foreground="black"
         ,font=("Arial",12))
    l13.grid(row=13,column=0,pady=10)
    
    e13=Entry(frame1,width=10,font=('Arial',15,"bold"),bg="lightgrey",fg="black"
                ,borderwidth=3)
    e13.grid(row=13,column=1,pady=10)

#button
    data_label=Label(root,text="",bg="lightblue",font=("Arial",14,"bold"))
    data_label.pack()
    def show_entry():
        p1=int(e1.get())
        p2=int(e2.get())
        p3=int(e3.get())
        p4=int(e4.get())
        p5=int(e5.get())
        p6=int(e6.get())
        p7=int(e7.get())
        p8=int(e8.get())
        p9=int(e9.get())
        p10=float(e10.get())
        p11=int(e11.get())
        p12=int(e12.get())
        p13=int(e13.get())
        model= joblib.load("Model_heart")
        result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
        
        if result == 0:
            data_label.config( text = "Cannot have Heart Diseases")
            
        else:
            data_label.config( text = "Can have Heart Diseases")

        

    
    button=Button(text="Predict",bg="lightblue",activebackground="lightgrey",
           borderwidth=3,command=show_entry)
    button.pack()



    

    root.mainloop()
except tkinter.TclError as e:
    print(f"Error: {e}")


# In[ ]:





# In[ ]:




