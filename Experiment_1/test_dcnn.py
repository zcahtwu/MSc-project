import torch
import os
from network import Net
import numpy as np
from torchvision import transforms
from network import Net
from loader import NewDataset,Loader
from math import ceil
from sklearn.metrics import auc, confusion_matrix,roc_curve
import matplotlib.pyplot as plt
import pickle

# define path
nth_try = 3
# print(f'seed = {nth_try -1}')
roc_path = f'./DCNNplot/ROC/ROC_plot{nth_try}'

# define test data path
test_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/Dataset/AbideValidSujit_Axial.h5'
test_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTestSujitNewNormalization_Axial.h5'
# transform validation and test image to tensor
other_transform = transforms.Compose([transforms.ToTensor()])
# define batch size
batch_size = 32
# define data loader and length of test set:
length_test, testloader = Loader(os.path.join(test_path),other_transform,batch_size,shuffle = False,num_workers = 4)
# define network
net = Net()
# load saved model
saved_model = f'./DCNNmodel/dcnn_balanced{nth_try}.pt'
# saved_model = f'./DCNNmodel/dcnn_sensitivity{nth_try}.pt'
# saved_model = 'with_t_best.pt'
net.load_state_dict(torch.load(saved_model))


# print test statistics
# prepare prediction and labels for testing data
test_scores = np.zeros(length_test)
test_preds = np.zeros(length_test)
test_labels = np.zeros(length_test)


# correctted_list = [76,85,105,132,145,147]
correctted_list = []
with torch.no_grad():
    # define the number of correct predicted samples
    n_correct = 0
    # loop over all test images
    for i, dataset in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        testimages, labels = dataset
        if i in correctted_list:
            labels = torch.zeros(32)
            # print(labels)
        # check cuda availability
        if torch.cuda.is_available():
            testimages = testimages.cuda()
            labels = labels.cuda()
            net = net.cuda() # send network to GPU
        # Get test score for each images in batch
        test_outputs = net(testimages)
        # make predictions
        predicted = [1 if torch.sigmoid(test_outputs[i])>= 0.5 else 0 for i in range(len(test_outputs))]
        batch_correct = [1 if predicted[i] == labels[i] else 0 for i in range(len(predicted))]
        # record all score, prediction and labels
        test_scores[i * batch_size : (i+1) * batch_size] = torch.sigmoid(test_outputs.cpu()[0:batch_size])
        test_preds[i * batch_size : (i+1) * batch_size] = predicted[0:batch_size]
        test_labels[i * batch_size : (i+1) * batch_size] = labels.cpu()[0:batch_size]
        # calcuate total test corrects
        n_correct += sum(batch_correct)
# define test accuracy
test_acc = 100.0 * n_correct / length_test

# plot ROC
fpr, tpr, thresholds = roc_curve(test_labels,test_scores)
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)
# plt.plot(fpr,tpr, label="ROC Curve",color="blue")
# plt.xlabel("False Positve Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.savefig(roc_path)
# plt.close()

# print accuracy
print('Test accuracy:%.2f'%test_acc)
# print confusion mateix:
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
print("==================")
print(confusion_matrix(test_labels,test_preds))
print("==================")
# sensitivity
sensitivity = tp/(tp+fn)
print(f'Sensitivity = {round(sensitivity,3)}')
# Specificity
specificity = tn/(tn + fp)
print(f'Specificity = {round(specificity,3)}')
# PPV
PPV = tp/(tp+fp)
print(f'PPV = {round(PPV,3)}')
# NPV
NPV = tn/(tn+fn)
print(f'NPV = {round(NPV,3)}')
# print AUC
auc_score = auc(fpr,tpr)
print('Auc = %.2f'%auc_score)
# print balanced acc
balanced_acc = round((sensitivity+specificity)/2,3)
print(f'Balanced_acc = {balanced_acc}')

# score_path = f'./DCNNsavedscore/DCNNscore_test{nth_try}.pkl'
# with open(score_path, 'wb') as f:
#     pickle.dump(test_scores, f)