import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Amjad Maaz\PycharmProjects\month_1\mushrooms.csv')

print(df.head())
encoder = LabelEncoder()
df = df.apply(encoder.fit_transform)

print(df.head())

X = df.drop(columns = ['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

print("X_train = ", X_train.shape)
print("y_train = ", y_train.shape)
print("X_test = ", X_test.shape)
print("y_test = ", y_test.shape)


def prior(y_train, label):
    total_points = y_train.shape[0]
    class_points = np.sum(y_train == label)

    return class_points / float(total_points)


def cond_prob(X_train, y_train, feat_col, feat_val, label):
    X_filtered = X_train[y_train == label]

    numerator = np.sum(X_filtered[feat_col] == feat_val)
    denominator = np.sum(y_train == label)

    return numerator / float(denominator)


def predict(X_train, y_train, xtest):
    # Get the number of target classes
    classes = np.unique(y_train)

    # All the features for our dataset
    features = [x for x in X_train.columns]

    # Compute posterior probabilites for each class
    post_probs = []

    for label in classes:
        likelihood = 1.0

        for f in features:
            cond = cond_prob(X_train, y_train, f, xtest[f], label)
            likelihood *= cond

        prior_prob = prior(y_train, label)

        posterior = prior_prob * likelihood

        post_probs.append(posterior)

        # Return the label for which the posterior probability was the maximum
    prediction = np.argmax(post_probs)

    return prediction

rand_example = 6

output = predict(X_train, y_train, X_test.iloc[rand_example])

print("Naive Bayes Classifier predicts ", output)
print("Current Answer ", y_test.iloc[rand_example])


def accuracy_score(X_train, y_train, xtest, ytest):
    preds = []

    for i in range(xtest.shape[0]):
        pred_label = predict(X_train, y_train, xtest.iloc[i])
        preds.append(pred_label)

    preds = np.array(preds)

    accuracy = np.sum(preds == ytest) / ytest.shape[0]

    return accuracy

print("Accuracy Score for our classifier == ", accuracy_score(X_train, y_train, X_test, y_test))
