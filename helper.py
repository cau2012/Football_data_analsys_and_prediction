import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys

class Helper:
    def Convert_to_tensor(self, data, XY):
        import torch
        from torch.autograd import Variable
        self.data = data
        if XY == 'X':
            tensor_data = torch.from_numpy(self.data)
            tensor_data = Variable(tensor_data)
            tensor_data = tensor_data.type(torch.FloatTensor)

        elif XY == 'Y':
            tensor_data = torch.from_numpy(self.data)
            tensor_data = Variable(tensor_data)
            tensor_data = tensor_data.type(torch.LongTensor)

        return tensor_data

    def Accuracy(self, model, X, Y):
        self.model = model
        self.Y = Y
        
        prediction = model(X)
        prediction = torch.nn.functional.softmax(prediction)
        correct_prediction = (torch.max(prediction.data, 1)[1] == Y.data)
        accuracy = correct_prediction.float().mean()

        return accuracy.item()

    def Evaluation(self, net, X_train1, y_train1, X_test1, y_test1, CNN=False):
        helper = Helper.data_loader()
        X_train_ = Convert_to_tensor(X_train1, 'X')
        Y_train_ = Convert_to_tensor(y_train1, 'Y')
        if CNN == True:
            X_train_ = X_train_.view(len(X_train1), 1, 10, 13)

        accuracy = Accuracy(net, X_train_, Y_train_)
        print('')
        print('-' * 60)
        print('Train Accuracy:', accuracy)

        X_test_ = Convert_to_tensor(X_test1, 'X')
        Y_test_ = Convert_to_tensor(y_test1, 'Y')
        if CNN == True:
            X_test_ = X_test_.view(len(X_test1), 1, 10, 13)

        accuracy = Accuracy(net, X_test_, Y_test_)
        print('Test Accuracy:', accuracy)

        single_prediction = net(X_test_)
        X_predict = torch.max(single_prediction.data, 1)[1]
        Y_predict = Y_test_.data

        X_predict_ = [int(i) for i in X_predict]
        Y_predict_ = [int(i) for i in Y_predict]

        match = []

        from sklearn.metrics import confusion_matrix
        predicted = confusion_matrix(X_predict_, Y_predict_)

        print('-' * 60)
        print(classification_report(X_predict_, Y_predict, target_names=['WIN', 'DRAW', 'LOSE']))
        print('-' * 60)
        nw, nd, nl = Y_predict_.count(0), Y_predict_.count(1), Y_predict_.count(2),
        print('Number of data : {}  W: {} /D: {} /L: {}'.format(len(Y_predict_), nw, nd, nl))
        print('-' * 60)
        print('Confusion Matrix')
        print(predicted)
        print('-' * 60)

    

    def Kprediction(self, net,x_data, loop=1000):
        def extract_predic_score(x_data):
            single_prediction = net(x_data)
            single_prediction.data
            return single_prediction.data
        self.net = net
        self.x_data = x_data
        prediction = net(x_data)
        holder = torch.zeros(prediction.data.size())
        for i in range(loop):
            holder += extract_predic_score(x_data)
            sys.stdout.write('\r{} / {}'.format(i + 1, loop))
        print('\n')
        holder = holder / loop

        X_predict = torch.max(holder, 1)[1]
        prob = torch.nn.functional.softmax(holder)
        return X_predict, prob