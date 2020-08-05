from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils import shuffle
import pickle

iris = load_iris()

X = iris.data
y = iris.target

X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

pickle.dump(classifier, open('ml_model.pkl','wb'))




