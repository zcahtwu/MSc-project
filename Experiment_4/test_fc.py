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
from sklearn.metrics import roc_curve, auc, roc_auc_score,balanced_accuracy_score
import sklearn

nth_dcnn = 3
# best_model = 98
correctted_list = [76,85,105,132,145,147]
correctted_list = []
# Define net
# The saved DCNN net
DCNN = f'./DCNNmodel/dcnn_balanced{nth_dcnn}.pt'
# DCNN = f'./DCNNmodel/1switch2/dcnn_epoch{295}.pt'
# DCNN = f'./DCNNmodel/dcnn_auc{nth_dcnn}.pt'
DNNnet = Net()
DNNnet.load_state_dict(torch.load(DCNN))

# set to evaluation model
DNNnet.eval()


# Check if Cuda is available
if torch.cuda.is_available():
    DNNnet = DNNnet.cuda()

# define test data path
test_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTestNewENAxial.h5'
test_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideValidNewENAxial.h5'
# define transform (to tensor)
other_transform = transforms.Compose([transforms.ToTensor()])
# define batch size
batch_size = 32
# define data loader and length of test set:
length_test, testloader = Loader(os.path.join(test_path),other_transform,batch_size,shuffle = False,num_workers = 4)

# calcuate number of batches of each set
n_batch_test = ceil(length_test/batch_size)

# Define the DCNN output and the label of test set
# No of slices in each volume
n_slices = 32
# for test set
test_all_score = np.zeros((int(length_test/n_slices), n_slices))
test_all_labels = np.zeros((int(length_test/n_slices),1))


# test data prepare
for i, dataset in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    data, labels = dataset
    if i in correctted_list:
        labels = torch.zeros(32)
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    # get DCNN output scores
    outputs = DNNnet(data)
    # record score and labels
    test_all_score[i,:] = outputs.cpu().detach().numpy()[0:n_slices]
    test_all_labels[i,0] = labels[0]
# convert score back into np_array
test_all_score = torch.from_numpy(test_all_score).float()
test_all_score = torch.sigmoid(test_all_score)
test_all_labels = torch.from_numpy(test_all_labels).float()


# create list for records
# nth_try = 10
nth_acc_list = []
nth_balanced_acc_list = []
nth_tn_list = []
nth_tp_list = []
nth_fp_list = []
nth_fn_list = []
sensitivity_list = []
specificity_list = []
NPV_list = []
PPV_list = [] 
auc_list = []
# test for each model
for nth_try in range(1,101):
    # define path
    # print(f'seed = {nth_try -1}')
    # roc_path = f'./FCplot/ROC/ROC_plot{nth_try}'

    # The saved FC net
    FC = f'./FCmodel/{nth_dcnn}r/balance{nth_try}.pt'
    net = FCnet()
    net.load_state_dict(torch.load(FC))
    net.eval()
    # Check if Cuda is available
    if torch.cuda.is_available():
        net = net.cuda()
    # calcuate relative metrics
    # define the number of correct predicted samples and total samples
    n_correct = 0
    n_samples = 0
    # define the FC output score list and prediction list
    score_list = np.zeros(int((length_test/n_slices)))
    pred_list = np.zeros(int((length_test/n_slices)))
    tp_list =[]

    # loop over all test sample
    for j in range(int((length_test/n_slices))):
        data = test_all_score[j,:]
        label = test_all_labels[j,0].reshape(1)
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        # Get score for one volume
        outputs = net(data)
        # update total number of sample
        n_samples += 1
        # make prediction based on sigmoid
        if torch.sigmoid(outputs) >= 0.5:
            predicted = 1
        else: 
            predicted = 0
        # if prediction correst, add one to n_correct
        if predicted == label:
            n_correct += 1
            if predicted ==1:
                tp_list.append(j)
        # record score and prediction
        score_list[j]=torch.sigmoid(outputs)
        pred_list[j] = predicted
    # score_path = f'./FCsavedscore/fc_test{nth_dcnn}.pkl'
    # if nth_try == best_model:
    #     with open(score_path, 'wb') as f:
    #         pickle.dump(score_list, f)
    # calcuate accuracy
    test_acc = 100.0 * n_correct / n_samples
    nth_acc_list.append(round(test_acc,3))
    # print(f'test accuracy = {round(test_acc,3)}')
    # record tn fp fn tp from confusion matrix and print confudion mateix
    # print(sklearn.metrics.confusion_matrix(test_all_labels,pred_list))
    tn, fp, fn, tp =sklearn.metrics.confusion_matrix(test_all_labels,pred_list).ravel()
    nth_fp_list.append(fp)
    nth_fn_list.append(fn)
    nth_tn_list.append(tn)
    nth_tp_list.append(tp)
    # sensitivity
    sensitivity = tp/(tp+fn)
    sensitivity_list.append(round(sensitivity,3))
    # print(f'Sensitivity = {round(sensitivity,3)}')
    # Specificity
    specificity = tn/(tn + fp)
    specificity_list.append(round(specificity,3))
    # print(f'Specificity = {round(specificity,3)}')
    # Balanced accuracy:
    nth_balanced_acc_list.append(round((sensitivity+specificity)/2,3))
    # PPV
    PPV = tp/(tp+fp)
    PPV_list.append(round(PPV,3))
    # print(f'PPV = {round(PPV,3)}')
    # NPV
    NPV = tn/(tn+fn)
    NPV_list.append(round(NPV,3))
    # print(f'NPV = {round(NPV,3)}')
    # plot ROC
    fpr, tpr, thresholds = roc_curve(test_all_labels,score_list)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)
    # plt.plot(fpr,tpr, label="ROC Curve",color="blue")
    # plt.xlabel("False Positve Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend()
    # plt.savefig(roc_path)
    # plt.close()
    # # calcuate AUC
    auc_score = auc(fpr,tpr)
    # print('Auc = %.2f'%auc_score)
    auc_list.append(round(auc_score,4))
    # print(f'TP list:{tp_list}')

print('# Accuracy:')
print('Accuracy = ',nth_acc_list)
print('Accuracy = [i/100 for i in Accuracy ]')
print('# True positive:')
print('tp = ',nth_tp_list)
print('# True negative:')
print('tn = ', nth_tn_list)
print('# False positive:')
print('fp = ', nth_fp_list)
print('# False negative:')
print('fn = ', nth_fn_list)
print('# Sensitivity:')
print('sensitivity = ', sensitivity_list)
print('# Specificity:')
print('specificity = ', specificity_list)
print('# PPV list')
print('ppv = ', PPV_list)
print('# NPV list')
print('npv = ' ,NPV_list)
print('# AUC')
print('auc = ', auc_list)
print('# Balanced accuracy:')
print('b_acc = ', nth_balanced_acc_list)
print()
print( f'# Accuracy:{round(np.mean(nth_acc_list)/100,3)} +_ {round(np.std(nth_acc_list)/100,3)}')
print( f'# Sensitivity:{round(np.mean(sensitivity_list),3)} +_ {round(np.std(sensitivity_list),3)}')
print( f'# Specificity:{round(np.mean(specificity_list),3)} +_ {round(np.std(specificity_list),3)}')
print( f'# ppv:{round(np.mean(PPV_list),3)} +_ {round(np.std(PPV_list),3)}')
print( f'# npv:{round(np.mean(NPV_list),3)} +_ {round(np.std(NPV_list),3)}')
print( f'# AUC:{round(np.mean(auc_list),3)} +_ {round(np.std(auc_list),3)}')
print( f'# Balanced accuracy:{round(np.mean(nth_balanced_acc_list),3)} +_ {round(np.std(nth_balanced_acc_list),3)}')
print(max(nth_balanced_acc_list), 1 + nth_balanced_acc_list.index(max(nth_balanced_acc_list)))