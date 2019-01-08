#!/bin/env python3.6
import os
import sys
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

"""
SVM
The main reason to use an SVM instead is because the problem might not be linearly separable. In that case, we will have
to use an SVM with a non linear kernel (e.g. RBF). Another related reason to use SVMs is if you are in a highly dimensional
space. For example, SVMs have been reported to work better for text classification. But it requires a lot of time for
training. So, it is not recommended when we have a large number of training examples.

kNN
It is robust to noisy training data and is effective in case of large number of training examples.
But for this algorithm, we have to determine the value of parameter K (number of nearest neighbors) and the type of
distance to be used.  The computation time is also very much as we need to compute distance of each query instance to
 all training samples.

Random Forest
Random Forest is nothing more than a bunch of Decision Trees combined. They can handle categorical features very well.
This algorithm can handle high dimensional spaces as well as large number of training examples.
Random Forests can almost work out of the box and that is one reason why they are very popular

Based on the above explanation,  I chose Random Forest, which is popular and works better for Text Classification!
"""

path = os.getcwd()


#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#
#%!% -------------Function to load the data and split it to X, y1 and y2!-------------------- %!%!#
#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#

def load_dataset(filename):
	"""this function loads the datafiles"""
	if not os.path.exists(filename):
		sys.exit('\n.... Oops ERROR: There is NO data file !?!?!?    \n')
	with open(path + '/' + filename, 'r', encoding="ISO-8859-1") as f:
		file = f.readlines()
		f.close()

	y1 = []
	y2 = []
	# y = [[],[]]
	X = []
	for line in file:
		line = line.replace(':', ' ')
		line = line.strip().split()

		# y1.append(line[0])
		# y2.append(line[1])
		y1.append(line[0])
		y2.append(line[1])
		X.append(' '.join(line[2:]))
	return X, y1, y2


#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#
#%!% -----------------Function to clean, stemm and lemattize the text!----------------------- %!%!#
#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#

def cleaning_stemming_lemattzing_text(text_array):
	documents = []
	for sen in range(0, len(text_array)):
		"""
		Applies some pre-processing on the given text.

		Steps :
		- Removing all the special characters
		- Removing all single characters
		- Removing single characters from the start
		- Substituting multiple spaces with single space
		- Removing prefixed 'b'
		- Converting to Lowercase
		- Lemmatization based on WordNet's morphy function
		- Stemming using the Porter Stemming Algorithm,
		"""
		document = re.sub(r'\W', ' ', str(text_array[sen]))   # Remove all the special characters
		document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)   # remove all single characters
		document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)    # Remove single characters from the start
		document = re.sub(r'\s+', ' ', document, flags=re.I)  # Substituting multiple spaces with single space
		document = document.lower()                   		  # Converting to Lowercase
		document = document.split()                           # Lemmatization and stemming
		document = [porter_stemmer.stem(wordnet_lemmatizer.lemmatize(word)) for word in document]
		document = ' '.join(document)
		documents.append(document)
	return documents


#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#
#%!% -----------Vectorizationo of the text and Training the model base on TDIFV!------------- %!%!#
#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#

X, y1, y2 = load_dataset('training.data')
documents = cleaning_stemming_lemattzing_text(X)

print('Starting Vectorization')
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.65, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()
# print(tfidfconverter.get_feature_names())

for ind, y in enumerate([y1, y2]):
# from sklearn.preprocessing import MultiLabelBinarizer
# binarizer = MultiLabelBinarizer()
	print('Starting Splitting Train_TEST')
	# dividing data into 20% test set and 80% training set.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	#######
	print('Starting Fitting Model')
	# 'string%s' % x
	# classifier_ind = 'classifier%s' %namestr(y)
	# globals()["classifier" + str(y)]
	classifier = RandomForestClassifier(n_estimators=100, random_state=0)
	# classifier.fit(X_train, y_train)
	classifier.fit(X_train, y_train)

	print('Starting Prediction!!')
	y_pred = classifier.predict(X_test)
	globals()["classifier" + str(ind)] = classifier



#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#
#%!% -------------------This part is for checking accuracy of the model!--------------------- %!%!#
#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#

	print('Starting test Score!!!')
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred))
	print(accuracy_score(y_test, y_pred))




#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#
#%!% ------------------------------ Prediction on new data----------------------------------- %!%!#
#%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%#


X_test_file, y1_test_file, y2_test_file = load_dataset('test.data')
documents_test_file = cleaning_stemming_lemattzing_text(X_test_file)
X_test_file = tfidfconverter.transform(documents_test_file).toarray()

for clf in [classifier0, classifier1]:
    y_pred_file = clf.predict(X_test_file)
    if clf == classifier0:
        print(classification_report(y1_test_file,y_pred_file))
        print('the prediction accuracy for the test file is {}% for lable 1'.format(float(accuracy_score(y1_test_file, y_pred_file))*100.0))
    if clf == classifier1:
        print(classification_report(y2_test_file,y_pred_file))
        print('the prediction accuracy for the test file is {}% for lable 2'.format(float(accuracy_score(y2_test_file, y_pred_file))*100.0))
