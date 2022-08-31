from sklearn.metrics import accuracy_score

def output_compute(y_pred):
    output = []
    for i in range(len(y_pred)):
        if max(y_pred[i]) == y_pred[i][0]:
            output.append(0.0)
        else:
            output.append(1.0)
    return output

def test(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    y_pred = []
    y_true = []

    for inputs, labels in testloader:
        output = model.forward(inputs)
        test_loss += criterion(output, labels.long()).item()

        y_pred += output_compute(output.tolist())
        y_true += labels.tolist()

    accuracy = (accuracy_score(y_true, y_pred))
    return test_loss, accuracy