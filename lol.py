import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from six import StringIO
from IPython.display import Image
import pydotplus
import os
import matplotlib.pyplot as mp #display picture
import matplotlib.image as mpimg #reading picture

# upload the date 
col_name=['gameld','creation','duration','seasonld','first','ftower','flnhibitor','fbaron','fdragon','friftherald','towerkills1','tinhiitorkills1','baronkills1','dragonkills1','riftheraldkills1','towerkills2','tinhiitorkills2','baronkills2','dragonkills2','riftheraldkills2','label']
pima=pd.read_cvs("test_set.cvs",header=None,names=clo_names)

pima=pima.iloc[1:]
pima.head()
#deal with feature and split the date by the features
feature_clos=['duration','first','ftower','flnhibitor','fbaron','fdragon','friftherald','towerkills1','tinhiitorkills1','baronkills1','dragonkills1','riftheraldkills1','towerkills2','tinhiitorkills2','baronkills2','dragonkills2','riftheraldkills2']
X=pima[feature_clos]
y=pima.label
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.6,random_state=1)
clf=DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
y_predict=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
#draw picture
os.environ["PATH"]+=os.pathsep+'c:/users/24391/anaconda3/lib/site-packages'
dot_date=StringIO()
export_graphviz(clf, out_file=dot_data,
 filled=True, rounded=True,
                special_characters=True,feature_names =
feature_cols,class_names=['1','2'])
graph=pydotplus.graph_form_dot_data(dot_data.getvalue())
graph.write_png('lol.png')
clf=DecisionTreeClassifier(critersion="entropy",max_depth=10)
clf=clf.fit(X_train,y_train)
y_predict=clf(X_test)
print("Accuracy2:",accuracy_score(y_test, y_pred))
dot_date=StringIO()
export_graphviz(clf,out_file=dot_data,
                filled=True,rounded=True,
                special_characters=True,featrue_names=
feature_clos,class_names=['1','2'])
graph=pydotplus,graph_from_dot_data(dot_data,getvalue())
graph.write_png('lol.png')
Image(graph.creat_png())
mylolanaly=mping.imread('lol.png')
plt.imshow(mylolanaly)
plt.axis('off')
plt.show()