# import packages
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
import random
from math import ceil
from torchvision import transforms
from network import Net, FCnet
from loader import NewDataset,Loader
from sklearn.metrics import balanced_accuracy_score,accuracy_score,recall_score, auc, roc_curve
# nth try of the training 
# =======================================
# For reproducibility:
nth_dcnn = 3
# torch.manual_seed((nth_try -1))
# random.seed(1)
np.random.seed(0)
# Disabling the benchmarking feature
torch.backends.cudnn.benchmark = False
# Avoiding nondeterministic algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# =======================================
# Define net:
# load the saved model for DCNN from saved_dcnn
saved_dcnn = f'./DCNNmodel/dcnn_balanced{nth_dcnn}.pt'
# saved_dcnn = f'./DCNNmodel/1switch2/dcnn_epoch{295}.pt'
# saved_dcnn = f'./DCNNmodel/dcnn_auc{nth_dcnn}.pt'
DNNnet = Net()
DNNnet.load_state_dict(torch.load(saved_dcnn))

# set DNN to evaluation mode
DNNnet.eval()

# Check if Cuda is available
if torch.cuda.is_available():
    # print("using GPU")
    # print("==================")
    DNNnet = DNNnet.cuda()

# load data set
# define training data path
train_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTrainNewENAxial.h5'
# define validation data path
val_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideValidNewENAxial.h5'
# test_path = r'AbideTestV1.h5'

# define augmentation trandformation for training data
torch.manual_seed(822)
augment_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomAffine(degrees=10,translate=(0.1,0.1)),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomResizedCrop(size=(256,256),scale = (0.8,1.2))
                            ])
# transform validation and test image to tensor
other_transform = transforms.Compose([transforms.ToTensor()])
# define batch size
batch_size = 32

# define data loader for each set and save the size of each data set as length
length_train, trainloader = Loader(os.path.join(train_path),augment_transform,batch_size, shuffle = False, num_workers = 4)
length_val, valloader = Loader(os.path.join(val_path),other_transform,batch_size,shuffle = False,num_workers = 4)
# length_test, testloader = Loader(os.path.join(test_path),other_transform,batch_size,shuffle = False,num_workers = 4)

# calcuate number of batches of each set
n_batch_train = ceil(length_train/batch_size)
n_batch_val = ceil(length_val/batch_size)
# n_batch_test = ceil(length_test/batch_size)

# Define the DCNN output and the label of each set
# No of slices in each volume
n_slices = 32
# For training set
train_all_score = np.zeros((int(length_train/n_slices), n_slices))
train_all_labels = np.zeros((int(length_train/n_slices),1))
# For validation set
val_all_score = np.zeros((int(length_val/n_slices), n_slices))
val_all_labels = np.zeros((int(length_val/n_slices),1))
# for test set
# test_all_score = np.zeros((int(length_test/n_slices), n_slices))
# test_all_labels = np.zeros((int(length_test/n_slices),1))

# train data prepare
for i, dataset in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    data, labels = dataset
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    # get DCNN output scores
    outputs = DNNnet(data)
    train_all_score[i,:] = outputs.cpu().detach().numpy()[0:n_slices]
    train_all_labels[i,0] = labels[0]
    # last batch may be size n_slices
# convert score back into np_array
train_all_score = torch.from_numpy(train_all_score).float()
train_all_score = torch.sigmoid(train_all_score)
train_all_labels = torch.from_numpy(train_all_labels).float()
# print(train_all_score.shape)
# val data prepare
for i, dataset in enumerate(valloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    data, labels = dataset
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    # get DCNN output scores
    outputs = DNNnet(data)
    val_all_score[i,:] = outputs.cpu().detach().numpy()[0:n_slices]
    val_all_labels[i,0] = labels[0]
# convert score back into np_array
val_all_score = torch.from_numpy(val_all_score).float()
val_all_score = torch.sigmoid(val_all_score)
val_all_labels = torch.from_numpy(val_all_labels).float()

# =======================================
for nth_try in range(1,11):
    # set seeds
    torch.manual_seed((nth_try))
    # define FC network
    net = FCnet()
    if torch.cuda.is_available():
        net = net.cuda()
    # Define path etc
    # print("==================")
    print(f'The {nth_try} th repeats, seed = {nth_try}')

    # saved_model_name1 = f'./FCmodel/accfc{nth_try}.pt'
    saved_model_name2 = f'./FCmodel/{nth_dcnn}drop/balance{nth_try}.pt'
    # saved_model_name1 = f'./FCmodel/{nth_dcnn}/auc{nth_try}.pt'
    # loss_plot_name = f'./FCplot/Loss/Loss_plot{nth_try}'
    # acc_plot_name = f'./FCplot/ACC/ACC_plot{nth_try}'


    # Training:
    # print("start training:")
    # Define the loss function weight the loss of positive sample
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(7.07))
    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # record the time that training start
    start_time = time.time()
    # create empty list for recording train and validation loss
    train_loss_list= []
    val_loss_list = []
    # create empty list for recording training and validation accuracy
    train_acc_list = []
    train_balanced_acc_list=[]
    train_auc_list = []
    val_acc_list = []
    val_balanced_acc_list = []
    val_sensitivity_list = []
    val_auc_list = []
    # define training loop
    n_epoch = 50
    # pick the highest validaton balanced accuracy for model selection:
    val_ba_model_list = []
    for epoch in range(n_epoch):
        train_pred_list=np.zeros(int((length_train/n_slices)))
        # train_label_list = np.zeros(int((length_test/n_slices)))
        train_score_list = np.zeros(int((length_train/n_slices)))
        val_pred_list = np.zeros(int((length_val/n_slices)))
        # val_label_list = []
        val_score_list = np.zeros(int((length_val/n_slices)))
        # record start time of each epoch
        epoch_start = time.time()
        # define and zero out training loss for each epoch
        training_loss = 0.0
        # enable train model for network
        net.train()
        # loop over all samlpe
        random_list = [i for i in range(0,int((length_train/n_slices)))]
        random.shuffle(random_list)
        # print(random_list)
        # for j in random_list:
        for j in range(int((length_val/n_slices))):
            data = train_all_score[j,:]
            labels = train_all_labels[j,0].reshape(1)
            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Apply forward function
            outputs = net(data)
            # calculate loss
            loss = criterion(outputs, labels)
            # Apply backward
            loss.backward()
            # Optimize
            optimizer.step()
            training_loss += loss.item()

        # enable evaluation mode
        net.eval()
        # calcuate performance, no gradient update involved 
        with torch.no_grad():
            for j in range(int((length_train/n_slices))):
                data = train_all_score[j,:]
                label = train_all_labels[j,0].reshape(1)
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                # get score of sample
                outputs = net(data)
                # make prediction based on score
                if torch.sigmoid(outputs) >= 0.5:
                    predicted = 1
                else: 
                    predicted = 0
                train_pred_list[j] = predicted
                # train_label_list.append(int(label.item()))
                train_score_list[j] = torch.sigmoid(outputs)
        # record training loss
        train_loss_list.append(training_loss/(length_train/n_slices))
        # calcuate and record training accuracy
        train_acc = accuracy_score(train_all_labels,train_pred_list)
        train_balanced_acc = balanced_accuracy_score(train_all_labels,train_pred_list)
        train_acc_list.append(train_acc)
        train_balanced_acc_list.append(train_balanced_acc)

        fpr, tpr, thresholds = roc_curve(train_all_labels,train_score_list)
        train_auc_score = auc(fpr,tpr)
        train_auc_list.append(train_auc_score)
        # validate:
        with torch.no_grad():
            # zero out validation loss for each epoch
            val_loss = 0

            # loop over all validation samples
            for j in range(int((length_val/n_slices))):
                data = val_all_score[j,:]
                label = val_all_labels[j,0].reshape(1)
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                # get score of sample
                outputs = net(data)

                # make prediction based on score
                if torch.sigmoid(outputs) >= 0.5:
                    predicted = 1
                else: 
                    predicted = 0
                # update validation loss
                val_loss += criterion(outputs, label)
                val_pred_list[j] = predicted
                # val_label_list.append(int(label.item()))
                val_score_list[j] = torch.sigmoid(outputs.cpu())

        # Val accuarcy
        val_acc_list.append(accuracy_score(val_all_labels,val_pred_list))
        val_balanced_acc_list.append(balanced_accuracy_score(val_all_labels,val_pred_list))
        val_sensitivity_list.append(recall_score(val_all_labels,val_pred_list,pos_label=1))
        fpr, tpr, thresholds = roc_curve(val_all_labels,val_score_list)
        val_auc_score = auc(fpr,tpr)
        val_auc_list.append(val_auc_score)
        # record validation loss
        val_loss_list.append(val_loss/(length_val/n_slices))
        
        
        # if val_auc_list[epoch] >= max(val_auc_list):
            # # print(val_auc_list[epoch])
            # torch.save(net.state_dict(), saved_model_name1)
     
        if val_balanced_acc_list[epoch] >= max(val_balanced_acc_list):
            # print(val_balanced_acc_list[epoch])
            torch.save(net.state_dict(), saved_model_name2)
        

        # record epoch end time
        epoch_end = time.time()
        # epoch time taken
        t_epoch = epoch_end - epoch_start
        # print epoch statistics
        trainacc = round(train_acc_list[epoch],2)
        trainbalanceacc = round(train_balanced_acc_list[epoch],2)
        valacc = round(val_acc_list[epoch],2)
        valbalanceacc = round(val_balanced_acc_list[epoch],2)
        trainloss = round(training_loss,2)
        valloss = round(val_loss.item(),2)
        # print(f'Epoch{epoch+1}: Train acc: {trainacc},train balanced acc:{trainbalanceacc}, Val acc:{valacc}, Val balanced acc:{valbalanceacc}, Train loss:{trainloss},Val loss = {valloss}')
    # record training end time
    end_time = time.time()
    # training total time taken
    t_all = end_time - start_time
    # print total time
    # print('training finish, total time: %.3f s' %(t_all))
    # val_ba_model_list.append(max(val_balanced_acc_list))
    # print the final model chosen
    picked_epoch1 = val_auc_list.index(max(val_auc_list))+1
    val_best_auc = round(max(val_auc_list),6)
    picked_epoch2 = val_balanced_acc_list.index(max(val_balanced_acc_list))+1
    val_best_ba = round(max(val_balanced_acc_list),4)
    print(f'Pick epoch {picked_epoch1} and {picked_epoch2} as final model, val auc{val_best_auc}, B acc = {val_best_ba}.')
# best_val_model = val_ba_model_list.index(max(val_ba_model_list))+1
# print(val_ba_model_list)
# print(best_val_model)

    # # plot loss changes
    # epoch_list = [i for i in range(0,n_epoch)]
    # val_loss_list  = torch.tensor(val_loss_list, device = 'cpu')
    # train_loss_list  = torch.tensor(train_loss_list, device = 'cpu')
    # plt.plot(epoch_list,val_loss_list,'b--',label ='val')
    # plt.plot(epoch_list,train_loss_list,'r',label = 'train')
    # plt.legend(loc=0)
    # plt.savefig(loss_plot_name)
    # plt.close()

    # # plot accuracy changes
    # val_acc_list  = torch.tensor(val_acc_list, device = 'cpu')
    # train_acc_list  = torch.tensor(train_acc_list, device = 'cpu')
    # plt.plot(epoch_list,val_acc_list,'b--',label ='val')
    # plt.plot(epoch_list,train_acc_list,'r',label = 'train')
    # plt.legend(loc=0)
    # plt.savefig(acc_plot_name)
    # plt.legend()
    # plt.close()
