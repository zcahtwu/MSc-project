# =======================================
# import library and other files
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torchvision import transforms
from network import Net
from loader import NewDataset,Loader
from math import ceil
from sklearn.metrics import balanced_accuracy_score,accuracy_score, recall_score, auc, confusion_matrix,roc_curve

nth_try = 1
# =======================================
# For reproducibility:
# Disabling the benchmarking feature
torch.backends.cudnn.benchmark = False
# Avoiding nondeterministic algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# set seeds
seeds = 28
# torch.manual_seed(3)
# torch.manual_seed(822)
# torch.manual_seed(31)
torch.manual_seed(seeds)
np.random.seed(1)
# =======================================
# Define path etc
print(f'seed = {seeds}')
# saved_model_name1 = f'./DCNNmodel/dcnn_acc{nth_try}.pt'
saved_model_name2 = f'./DCNNmodel/dcnn_balanced{nth_try}.pt'
saved_model_name3 = f'./DCNNmodel/dcnn_auc{nth_try}.pt'
loss_plot_name = f'./DCNNplot/Loss/Loss_plot{nth_try}'
acc_plot_name = f'./DCNNplot/ACC/ACC_plot{nth_try}'
current_folder = 'new_intensity_axial'
# Define net
net = Net()

# Check if Cuda is available
if torch.cuda.is_available():
    print("Using GPU")
    print("==================")
    net = net.cuda() # send network to GPU

# Load data set
# define training data path
train_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTrainNewENAxial.h5'
# define validation data path
val_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideValidNewENAxial.h5'
# define test data path
test_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTestNewENAxial.h5'

# define augmentation trandformation for training data
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
batch_size = 64
# define data loader for each set and save the size of each data set as length
length_train, trainloader = Loader(os.path.join(train_path),augment_transform,batch_size, shuffle = True, num_workers = 8)
length_val, valloader = Loader(os.path.join(val_path),other_transform,batch_size,shuffle = False,num_workers = 8)
length_test, testloader = Loader(os.path.join(test_path),other_transform,32,shuffle = False,num_workers = 4)

# calcuate number of batches of each set
n_batch_train = ceil(length_train/batch_size)
n_batch_val = ceil(length_val/batch_size)
n_batch_test = ceil(length_test/batch_size)

# =======================================
# Train:
print("start training:")
# Define the loss function weight the loss of positive sample
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(7.07))
# define the un-weighted loss
criterion2 = torch.nn.BCEWithLogitsLoss()
# define optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
# record the time that training start
start_time = time.time()
# create empty list for recording train and validation loss
train_loss_list= []
train_non_weighted_loss_list = []
val_loss_list = []
val_non_weighted_loss_list = []
test_loss_list = []
test_non_weighted_loss_list = []
# create empty list for recording training and validation accuracy
train_acc_list = []
train_balanced_acc_list=[]
train_sensitivity_list=[]
train_auc_list = []
val_acc_list = []
val_balanced_acc_list = []
val_sensitivity_list=[]
val_auc_list = []
test_acc_list = []
test_balanced_acc_list = []
test_sensitivity_list=[]
test_auc_list = []
#
# Define training loop
n_epoch = 500
for epoch in range(n_epoch):
    train_pred_list =  np.zeros((int(length_train),1))
    train_label_list = np.zeros((int(length_train),1))
    train_score_list = np.zeros((int(length_train)))
    val_pred_list =  np.zeros((int(length_val),1))
    val_label_list = np.zeros((int(length_val),1))
    val_score_list = np.zeros((int(length_val)))
    test_pred_list =  np.zeros((int(length_test),1))
    test_label_list = np.zeros((int(length_test),1))
    test_score_list = np.zeros((int(length_test)))
    # record start time of each epoch
    epoch_start = time.time()
    # define and zero out training loss for each epoch
    training_loss = 0.0
    train_non_weighted_loss = 0.0
    # enable train model for network
    net.train()

    # loop over each batch
    for i, dataset in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = dataset
        # check gpu and send data into gpu
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # Apply forward function
        outputs = net(inputs)
        # calculate batch loss
        loss = criterion(outputs, labels)
        # Apply backward
        loss.backward()
        # Optimize
        optimizer.step()
        # add batch loss to training loss
        training_loss += loss.item()
        train_non_weighted_loss += criterion2(outputs, labels).item()

    # record average sample training loss into list
    train_loss_list.append(training_loss/n_batch_train)
    train_non_weighted_loss_list.append(train_non_weighted_loss/n_batch_train)
    # enable evaluation mode
    net.eval()
    # calcuate performance, no gradient update involved 
    with torch.no_grad():
        # loop over all training samples
        for i, dataset in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = dataset

            # check gpu and send data into gpu
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # get score of sample in one batch
            outputs = net(inputs)
            # make prediction by apply sigmoid to score
            predicted = [1 if torch.sigmoid(outputs[j])>= 0.5 else 0 for j in range(len(outputs))]
            train_score_list[i * batch_size : (i+1) * batch_size] = torch.sigmoid(outputs.cpu()[0:batch_size])
            # record prediction and labels
            train_pred_list[64*i:(i+1)*64,0] = predicted[:]
            train_label_list[64*i:(i+1)*64,0] = labels.cpu().detach().numpy()[:]
        # convert training label to numpy array
        train_label_list = torch.from_numpy(train_label_list).float()

    # calcuate auc
    fpr, tpr, thresholds = roc_curve(train_label_list,train_score_list)
    train_auc_score = auc(fpr,tpr)
    train_auc_list.append(train_auc_score)
    # calcuate training accuracy 
    train_acc = accuracy_score(train_label_list,train_pred_list)
    # record training accuracy
    train_acc_list.append(train_acc)
    train_balanced_acc_list.append(balanced_accuracy_score(train_label_list,train_pred_list))
    train_sensitivity_list.append(recall_score(train_label_list,train_pred_list,pos_label=1))

    # print validation statistics
    with torch.no_grad():
        # zero out validation loss for each epoch
        val_loss = 0
        val_non_weighted_loss = 0

        for i, dataset in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            valimages, labels = dataset

            # check gpu and send data into gpu
            if torch.cuda.is_available():
                valimages = valimages.cuda()
                labels = labels.cuda()
            # get score of sample in one batch
            val_outputs = net(valimages)
            # # make prediction by apply sigmoid to score
            predicted = [1 if torch.sigmoid(val_outputs[j])>= 0.5 else 0 for j in range(len(val_outputs))]
            # record score
            val_score_list[i * batch_size : (i+1) * batch_size] = torch.sigmoid(val_outputs.cpu()[0:batch_size])
            # record prediction and label for validation   
            val_pred_list[64*i:(i+1)*64,0] = predicted[:]
            val_label_list[64*i:(i+1)*64,0] = labels.cpu().detach().numpy()[:]
            # add validation batch loss to total validation loss
            val_loss += criterion(val_outputs, labels)
            val_non_weighted_loss += criterion2(val_outputs, labels)

        # covert val label to np array
        val_label_list = torch.from_numpy(val_label_list).float()
    
    # calcuate auc
    fpr, tpr, thresholds = roc_curve(val_label_list,val_score_list)
    val_auc_score = auc(fpr,tpr)
    val_auc_list.append(val_auc_score)
    # record validation accuracy
    val_acc_list.append(accuracy_score(val_label_list,val_pred_list))
    val_balanced_acc_list.append(balanced_accuracy_score(val_label_list,val_pred_list))
    val_sensitivity_list.append(recall_score(val_label_list,val_pred_list,pos_label=1))
    # record validation loss
    val_loss_list.append(val_loss/n_batch_val)
    val_non_weighted_loss_list.append(val_non_weighted_loss/n_batch_val)

    # print Test statistics
    with torch.no_grad():
        correctted_list = [76,85,105,132,145,147]
        # zero out test loss for each epoch
        test_loss = 0
        test_non_weighted_loss = 0

        for i, dataset in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            testimages, labels = dataset
            if i in correctted_list:
                labels = torch.zeros(32)
            # check gpu and send data into gpu
            if torch.cuda.is_available():
                testimages = testimages.cuda()
                labels = labels.cuda()
            # get score of sample in one batch
            test_outputs = net(testimages)
            # # make prediction by apply sigmoid to score
            predicted = [1 if torch.sigmoid(test_outputs[j])>= 0.5 else 0 for j in range(len(test_outputs))]
            # record score
            test_score_list[i * 32 : (i+1) * 32] = torch.sigmoid(test_outputs.cpu()[0:32])
            # record prediction and label for test   
            test_pred_list[32*i:(i+1)*32,0] = predicted[:]
            test_label_list[32*i:(i+1)*32,0] = labels.cpu().detach().numpy()[:]
            # add test batch loss to total test loss
            test_loss += criterion(test_outputs, labels)
            test_non_weighted_loss += criterion2(test_outputs, labels)

        # covert test label to np array
        test_label_list = torch.from_numpy(test_label_list).float()
    
    # calcuate auc
    fpr, tpr, thresholds = roc_curve(test_label_list,test_score_list)
    test_auc_score = auc(fpr,tpr)
    test_auc_list.append(test_auc_score)
    # record test accuracy
    test_acc_list.append(accuracy_score(test_label_list,test_pred_list))
    test_balanced_acc_list.append(balanced_accuracy_score(test_label_list,test_pred_list))
    test_sensitivity_list.append(recall_score(test_label_list,test_pred_list,pos_label=1))
    # record test loss
    test_loss_list.append(test_loss/n_batch_test)
    test_non_weighted_loss_list.append(test_non_weighted_loss/n_batch_test)
    # # save model if the validation accuracy is the best
    # # if epoch+1 > 100:
    #     # if val_acc_list[epoch] >= max(val_acc_list[100:]):
    #     #     torch.save(net.state_dict(), saved_model_name1)

    # if val_balanced_acc_list[epoch] >= max(val_balanced_acc_list):#[100:]):
    #     torch.save(net.state_dict(), saved_model_name2)

    # if val_auc_list[epoch] >= max(val_auc_list):
    #     torch.save(net.state_dict(), saved_model_name3)
    # record epoch end time 
    all_model_name = f'./DCNNmodel/1switch2/dcnn_epoch{epoch}.pt'
    torch.save(net.state_dict(), all_model_name)
    epoch_end = time.time()
    # epoch time taken
    t_epoch = epoch_end - epoch_start

    # print epoch statistics
    # first round them
    trainacc = round(100*train_acc_list[epoch],2)
    auc_train = round(train_auc_list[epoch],4)
    trainbalanceacc = round(100*train_balanced_acc_list[epoch],2)

    valacc = round(100*val_acc_list[epoch],2)
    auc_val = round(val_auc_list[epoch],4)
    valbalanceacc = round(100*val_balanced_acc_list[epoch],2)

    testacc = round(100*test_acc_list[epoch],2)
    auc_test = round(test_auc_list[epoch],4)
    testbalanceacc = round(100*test_balanced_acc_list[epoch],2)

    trainloss = round(training_loss,2)
    valloss = round(val_loss.item(),2)
    testloss = round(test_loss.item(),2)

    trainsensiticity = round(train_sensitivity_list[epoch],3)
    valsensitivity = round(val_sensitivity_list[epoch],3)
    testsensitivity = round(test_sensitivity_list[epoch],3)

    print(f'Epoch[{epoch+1}] Train acc={trainacc}, Train b acc={trainbalanceacc}, Train sensitivity={trainsensiticity}, Train auc = {auc_train}  Train loss:{trainloss}')
    print(f'         Valid acc={valacc}, Valid b acc:{valbalanceacc}, Valid sensitivity={valsensitivity}, Valid auc = {auc_val},Valid loss = {valloss}')
    print(f'         Test acc={testacc}, Test b acc:{testbalanceacc}, Test sensitivity={testsensitivity}, Test auc = {auc_test},Test loss = {testloss}')
# record training end time
end_time = time.time()
# training total time taken
t_all = end_time - start_time
# print total time
print('training finish, total time: %.3f s' %(t_all))
# print the final model chosen
# picked_epoch1 = val_acc_list.index(max(val_acc_list[100:]))+1
# picked_epoch2 = val_balanced_acc_list.index(max(val_balanced_acc_list[100:]))+1
# print(f'Pick epoch {picked_epoch2} as final model.')

# plot loss changes
# epoch_list = [i for i in range(0,n_epoch)]
val_loss_list  = torch.tensor(val_loss_list, device = 'cpu')
val_non_weighted_loss_list = torch.tensor(val_non_weighted_loss_list,device = 'cpu')
train_loss_list  = torch.tensor(train_loss_list, device = 'cpu')
train_non_weighted_loss_list = torch.tensor(train_non_weighted_loss_list,device = 'cpu')
test_loss_list  = torch.tensor(test_loss_list, device = 'cpu')
test_non_weighted_loss_list = torch.tensor(test_non_weighted_loss_list,device = 'cpu')
# plt.plot(epoch_list,val_loss_list,'tab:blue',label ='val weighted')
# plt.plot(epoch_list,val_non_weighted_loss_list,'teal',label ='val non-weighted')
# plt.plot(epoch_list,train_loss_list,'tab:red',label = 'train weighted')
# plt.plot(epoch_list,train_non_weighted_loss_list,'lightsalmon',label = 'train non-weighted')
# plt.ylim(0,4)
# plt.xlim(0,500)
# plt.title('Loss track plot')
# plt.ylabel('Batch Loss')
# plt.xlabel('Number of Epoch')
# plt.legend(loc='upper left')
# plt.savefig(loss_plot_name)
# plt.close()

# # plot accuracy changes
val_acc_list  = torch.tensor(val_acc_list, device = 'cpu')
val_balanced_acc_list = torch.tensor(val_balanced_acc_list, device = 'cpu')
train_acc_list  = torch.tensor(train_acc_list, device = 'cpu')
train_balanced_acc_list = torch.tensor(train_balanced_acc_list, device = 'cpu')
test_acc_list  = torch.tensor(test_acc_list, device = 'cpu')
test_balanced_acc_list = torch.tensor(test_balanced_acc_list, device = 'cpu')
# plt.plot(epoch_list,val_acc_list,'tab:blue',label ='val acc')
# plt.plot(epoch_list,val_balanced_acc_list,'teal',label ='val balanced acc')
# plt.plot(epoch_list,train_acc_list,'tab:red',label = 'train acc')
# plt.plot(epoch_list,train_balanced_acc_list,'lightsalmon',label = 'train balanced acc')
# plt.ylim(0.2,1)
# plt.xlim(0,500)
# plt.title('Accuracy track plot')
# plt.ylabel('Accuracy')
# plt.xlabel('Number of Epoch')
# plt.legend(loc='lower right')
# plt.savefig(acc_plot_name)
# plt.close()


# save the loss and accuracy
# acc:
train_acc_path = f'./DCNNoutput/{nth_try}/train_acc{nth_try}.pkl'
with open(train_acc_path, 'wb') as f:
    pickle.dump(train_acc_list, f)

train_balanced_acc_path = f'./DCNNoutput/{nth_try}/train_balanced_acc{nth_try}.pkl'
with open(train_balanced_acc_path, 'wb') as f:
    pickle.dump(train_balanced_acc_list, f)

val_acc_path = f'./DCNNoutput/{nth_try}/val_acc{nth_try}.pkl'
with open(val_acc_path, 'wb') as f:
    pickle.dump(val_acc_list, f)

val_balanced_acc_path = f'./DCNNoutput/{nth_try}/val_balanced_acc{nth_try}.pkl'
with open(val_balanced_acc_path, 'wb') as f:
    pickle.dump(val_balanced_acc_list, f)   

test_acc_path = f'./DCNNoutput/{nth_try}/test_acc{nth_try}.pkl'
with open(test_acc_path, 'wb') as f:
    pickle.dump(test_acc_list, f)

test_balanced_acc_path = f'./DCNNoutput/{nth_try}/test_balanced_acc{nth_try}.pkl'
with open(test_balanced_acc_path, 'wb') as f:
    pickle.dump(test_balanced_acc_list, f)
# loss:
train_loss_path = f'./DCNNoutput/{nth_try}/train_loss{nth_try}.pkl'
with open(train_loss_path, 'wb') as f:
    pickle.dump(train_loss_list, f)

train_non_weighted_loss_path = f'./DCNNoutput/{nth_try}/train_non_weighted_loss{nth_try}.pkl'
with open(train_non_weighted_loss_path, 'wb') as f:
    pickle.dump(train_non_weighted_loss_list, f)

val_loss_path = f'./DCNNoutput/{nth_try}/val_loss{nth_try}.pkl'
with open(val_loss_path, 'wb') as f:
    pickle.dump(val_loss_list, f)

val_non_weighted_loss_path = f'./DCNNoutput/{nth_try}/val_non_weighted_loss{nth_try}.pkl'
with open(val_non_weighted_loss_path, 'wb') as f:
    pickle.dump(val_non_weighted_loss_list, f)   

test_loss_path = f'./DCNNoutput/{nth_try}/test_loss{nth_try}.pkl'
with open(test_loss_path, 'wb') as f:
    pickle.dump(test_loss_list, f)

test_non_weighted_loss_path = f'./DCNNoutput/{nth_try}/test_non_weighted_loss{nth_try}.pkl'
with open(test_non_weighted_loss_path, 'wb') as f:
    pickle.dump(test_non_weighted_loss_list, f)