

# Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#col_names = ['Truck_Model', 'PayLoad(High)', 'Co2_Emission(gram/Km)', 'starting_location', 'Destination']
# load dataset
pima = pd.read_excel("C:/Users/HARSHITH REDDY/Desktop/present project/Project_73_Data.xlsx")
pima.describe()
pima.isnull().sum()
pima.duplicated()
pima.head()
pima.info()
pima.columns

#split dataset in features and target variable
feature_cols = ['Payload', 'Type','Dist_To_Travel','region/non_region']
X = pima[feature_cols] # Features
y = pima['vehicleType'] # Target variable
k = pima['vehicleType']

######### Label Encoder ############
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

X['Type']= labelencoder.fit_transform(X['Type'])
X['Dist_To_Travel'] = labelencoder.fit_transform(X['Dist_To_Travel'])
X['region/non_region']= labelencoder.fit_transform(X['region/non_region'])


y=labelencoder.fit_transform(y)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#from sklearn.tree import export_graphviz
#from six import StringIO
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus #execute pip install pydotplus


#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#sudo apt-get install graphviz  
#graph.write_png('Trucks.png')
#Image(graph.create_png())

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#pip install graphviz 
#pip install pydotplus

# Visualizing Decision Trees
#from six import StringIO  
#from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True, feature_names = feature_cols,class_names=['0','1','2','3','4','5','6','7'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('Truck.png')
#Image(graph.create_png())


       
pickle.dump(clf, open('bigdatatree.pkl','wb'))
model = pickle.load(open('bigdatatree.pkl','rb'))
print(model.predict([[2000,0,1,1]]))




