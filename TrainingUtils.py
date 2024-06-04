import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from TestModel import TestModel
from PointResNet import PointResNet
from LoadPCData import *
from torch.utils.data import Dataset, DataLoader
import math
import time

def getRotation(batch, rotationAmount):
    # Return the rotate batch of point clouds
    # rotationAmount: rotation about the z-axis in radians.
    rotMatrix = torch.tensor([[math.cos(rotationAmount), -math.sin(rotationAmount), 0],[math.sin(rotationAmount), math.cos(rotationAmount), 0],[0,0,1]]);
    rotatedPoints = torch.matmul(batch.reshape((batch.shape[0], batch.shape[1], 1, batch.shape[2])).double(), rotMatrix.reshape((1,1,3,3)).double());
    rotatedPoints = rotatedPoints.reshape(batch.shape);
    return rotatedPoints

def expandDataSetUsingRotation(batch, labels, n):
    # This function will enlarge the dataset using rotations of 2*pi/n
    # The returned dataset will be (n)*larger
    largerDataSet = torch.clone(batch);
    largerLabelSet = torch.clone(labels);
    for divider in range(1,n):
        rotatedDataset = getRotation(batch, 2*divider*math.pi/(n));
        largerDataSet = torch.cat((largerDataSet, rotatedDataset), dim=0);
        largerLabelSet = torch.cat((largerLabelSet, labels), dim=0);
    return largerDataSet, largerLabelSet

def getTrainingWeights(trainingClasses, weights_exponent=0):
    frequencies = []
    for i in range(40):
        frequencies.append(torch.sum(trainingClasses == i));
    trainingWeights = torch.tensor(frequencies);
    trainingWeights = trainingWeights.max()/trainingWeights;
    return trainingWeights**weights_exponent



def train_model_and_save_data (model, epochs, learning_rate, weight_decay,
                               data_expansion_multiplier=1, weight_training_data = False,
                               data_name="training_data", model_name="best_model",
                               learning_rate_multiplier = 1, learning_rate_epoch_step = 1):
    # Gather training and validation data
    startTime = time.time();
    train_data, val_data, train_classes, val_classes = load_saved_pointclouds();

    # if we set data_expansion_multiplier, perform rotations on point clouds to create more training data
    # The sets or rotated data are 2*pi/i for i = 1:data_expansion_multiplier
    if data_expansion_multiplier > 1:
        train_data, train_classes = expandDataSetUsingRotation(train_data, train_classes, 10);

    if weight_training_data:
        training_weights = getTrainingWeights(train_classes, weights_exponent=.75);
    else:
        training_weights = torch.ones(40)

    loader = DataLoader(list(zip(train_data, train_classes)), shuffle=True, batch_size=32);
    val_loader = DataLoader(list(zip(val_data, val_classes)), shuffle=True, batch_size=32);
    # Train the model
    loss_fn = nn.CrossEntropyLoss(weight = training_weights);

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9, weight_decay=weight_decay);
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_epoch_step,
                                                gamma=learning_rate_multiplier);
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay);
    train_losses = [];
    train_accuracies = [];
    val_losses = [];
    val_accuracies = [];
    best_validation_accuracy = 0;
    num = 0
    for epoch in range(epochs):


        correct_training_predictions = 0;
        total_training_predictions = 0;
        total_training_loss = 0;
        batch_number = 0;

        # Train on one epoch
        for X_batch, y_batch in loader:
            print(num)
            num += 1
            # zero the parameter gradients
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            model.train()
            y_pred = model(X_batch.to(torch.float32))
            predictions = y_pred.argmax(dim=1);
            correct_training_predictions += torch.sum(predictions == y_batch).item();
            total_training_predictions += predictions.size()[0];

            #y_one_hot = torch.nn.functional.one_hot(y_batch, 40).to(torch.float32);
            loss = loss_fn(y_pred, y_batch)# * training_weights[y_batch]).mean();
            total_training_loss += loss.item();

            loss.backward()
            optimizer.step()

            batch_number += 1;

        print("-----------------Values For Epoch {} ---------------------".format(epoch));
        print("Training Accuracy: {}".format(correct_training_predictions/total_training_predictions));
        print("Training Loss: {}".format(total_training_loss/batch_number));
        train_losses.append(total_training_loss/batch_number);
        train_accuracies.append(correct_training_predictions/total_training_predictions);


        # Generate validation accuracy.
        model.eval()

        total_val_predictions = 0
        correct_val_predictions = 0
        total_loss = 0
        counter = 0
        for X_batch, y_batch in val_loader:

            y_pred = model(X_batch.to(torch.float32))
            predictions = y_pred.argmax(dim=1);
            correct_val_predictions += torch.sum(predictions == y_batch).item();
            total_val_predictions += predictions.size()[0];
            loss = loss_fn(y_pred, y_batch);
            total_loss += loss.item()
            counter += 1
        '''for i in range(40):
            class_pred_line = 'Class ' + str(i) + ' Predictions:'
            print(class_pred_line)
            print(torch.sum(predictions==i));'''


        val_accuracy = correct_val_predictions/total_val_predictions;
        print("\nValidation Accuracy: {}".format(correct_val_predictions/total_val_predictions));
        val_accuracies.append(correct_val_predictions/total_val_predictions)
        # Generate validation loss.
        #y_one_hot = torch.nn.functional.one_hot(val_classes, 40).to(torch.float32);

        print("Validation Loss: {}\n".format(total_loss/counter));
        val_losses.append(total_loss/counter);
        newTime = time.time();
        timeDiff = startTime - newTime;
        print("This Epoch took {} minutes\n\n".format(timeDiff/60));
        startTime = time.time();
        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy;
            torch.save(model, model_name);
        scheduler.step();
    return train_losses, train_accuracies, val_losses, val_accuracies;

def plotAndSave(trainingData, plotName, x_label, y_label, fileName):
    x_data = np.array(range(trainingData.size));
    '''fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x_data, trainingData)'''
    plt.plot(x_data, trainingData);
    plt.title(plotName);
    plt.xlabel(x_label);
    plt.ylabel(y_label);

    plt.savefig(fileName);
    plt.show();


def runTrainingExperiment(model, epochs, lr, modelName='DeepModel',
                          data_expansion_multiplier=1,
                          weight_training_data = False,
                          L2_norm_constant=0,
                          learning_rate_multiplier = 1,
                          learning_rate_epoch_step = 1):
    train_losses, train_accuracies, val_losses, val_accuracies = train_model_and_save_data(model, epochs, learning_rate=lr,
                                                                                            weight_decay=L2_norm_constant,
                                                                                            data_expansion_multiplier=data_expansion_multiplier,
                                                                                            weight_training_data=weight_training_data,
                                                                                            model_name=modelName, learning_rate_multiplier=learning_rate_multiplier,
                                                                                            learning_rate_epoch_step=learning_rate_epoch_step);
    plotAndSave(np.array(train_losses), modelName + ' Training Loss', 'Epoch', 'Training Loss', modelName + '_training_loss.png');
    plotAndSave(np.array(train_accuracies), modelName + ' Training Accuracy', 'Epoch', 'Training Accuracy', modelName + '_training_accuracy.png');
    plotAndSave(np.array(val_losses), modelName + ' Validation Loss', 'Epoch', 'Validation Loss', modelName + '_validation_loss.png');
    plotAndSave(np.array(val_accuracies), modelName + ' Validation Accuracy', 'Epoch', 'Validation Accuracy', modelName + '_validation_accuracy.png');


if __name__ == "__main__":

    model = PointResNet();
    runTrainingExperiment(model, 10, lr=.001, modelName='PointNet', weight_training_data=True, L2_norm_constant=.001, data_expansion_multiplier=1, learning_rate_epoch_step=8, learning_rate_multiplier=.5);
#train_model_and_save_data(model, 10, learning_rate=.01, model_name='VoxNetBest');
