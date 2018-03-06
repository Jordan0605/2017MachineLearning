from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
from sys import argv
from math import log, sqrt, pi, exp

def NB_classifier():
    n_features = len(x_train[0])
    #init
    classTable = [[[0]*256 for i in range(n_features)] for j in range(10)]
    #divide by class
    class_count = [0]*10
    for i in range(len(x_train)):
        c = y_train[i] #this class
        class_count[c] += 1
        for j in range(n_features):
            value = x_train[i][j]
            classTable[c][j][value] += 1
    #predict(Laplace is added to avoid 0)
    total = reduce(lambda x, y: x+y, class_count)
    pred = [[log(float(class_count[i]+1)/float(total+9)) for i in range(10)] for j in range(len(x_test))]
    for i in range(len(x_test)):
        for j in range(n_features):
            value = x_test[i][j]
            for k in range(10):
                pred[i][k] += log(float(classTable[k][j][value]+1)/float(class_count[k]+9))
    #find highest posterior
    pred_label = [0 for i in range(len(pred))]
    for i in range(len(pred)):
        predict_class = 0
        value = pred[i][0]
        for j in range(10):
            if pred[i][j] > value:
                value = pred[i][j]
                predict_class = j
        pred_label[i] = predict_class
    #output
    print confusion_matrix(y_test, pred_label)
    print accuracy_score(y_test, pred_label)

def MeanSD(arr):
    m = 0.0
    for element in arr:
        m += element
    m /= len(arr)
    sd = 0.0
    for x in arr:
        sd += pow(float(x)-m, 2)
    sd = sqrt(sd/(len(arr)-1))
    return m, sd

def GaussianNB():
    n_features = len(x_train[0])
    #init
    classTable = [[[] for i in range(n_features)] for j in range(10)]
    #divide by class
    for i in range(len(x_train)):
        c = y_train[i] #this class
        for j in range(n_features):
            value = x_train[i][j]
            classTable[c][j].append(value)
    #calculate Mean and Standard Deviation
    valueTable = [[] for i in range(10)]
    for i in range(10):
        for j in range(n_features):
            m, sd = MeanSD(classTable[i][j])
            valueTable[i].append((m, sd))
    #predict
    pred = [[0.0]*10 for i in range(len(x_test))]
    for i in range(len(x_test)):
        for j in range(n_features):
            v = x_test[i][j]
            for k in range(10):
                (m, sd) = valueTable[k][j]
                if sd == 0.0:
                    continue
                prob = log(1/(sqrt(2*pi)*sd)) - (pow(v-m, 2)/(2*pow(sd, 2)))
                pred[i][k] += prob
    #find highest posterior
    pred_label = [0 for i in range(len(x_test))]
    for i in range(len(x_test)):
        predict_class = 0
        value = pred[i][0]
        for j in range(10):
            if pred[i][j] > value:
                value = pred[i][j]
                predict_class = j
        pred_label[i] = predict_class
    #output
    print confusion_matrix(y_test, pred_label)
    print accuracy_score(y_test, pred_label)

    
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    
    if len(argv) > 1 and argv[1] == "1":
        GaussianNB()
    else:
        NB_classifier()
