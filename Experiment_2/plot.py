import matplotlib.pyplot as plt
import pickle
import torch
nth_try = 'new1'
n_epoch = 500

loss_plot_name = f'./DCNNplot/Loss/Loss_plot{nth_try}'
acc_plot_name = f'./DCNNplot/ACC/ACC_plot{nth_try}'


# acc:
train_acc_path = f'./DCNNoutput/{nth_try}/train_acc{nth_try}.pkl'
with open(train_acc_path, 'rb') as f:
    train_acc_list = pickle.load(f)

train_balanced_acc_path = f'./DCNNoutput/{nth_try}/train_balanced_acc{nth_try}.pkl'
with open(train_balanced_acc_path, 'rb') as f:
    train_balanced_acc_list = pickle.load(f)

val_acc_path = f'./DCNNoutput/{nth_try}/val_acc{nth_try}.pkl'
with open(val_acc_path, 'rb') as f:
    val_acc_list = pickle.load(f)

val_balanced_acc_path = f'./DCNNoutput/{nth_try}/val_balanced_acc{nth_try}.pkl'
with open(val_balanced_acc_path, 'rb') as f:
    val_balanced_acc_list = pickle.load(f)
# loss:
train_loss_path = f'./DCNNoutput/{nth_try}/train_loss{nth_try}.pkl'
with open(train_loss_path, 'rb') as f:
    train_loss_list = pickle.load(f)

train_non_weighted_loss_path = f'./DCNNoutput/{nth_try}/train_non_weighted_loss{nth_try}.pkl'
with open(train_non_weighted_loss_path, 'rb') as f:
    train_non_weighted_loss_list = pickle.load(f)

val_loss_path = f'./DCNNoutput/{nth_try}/val_loss{nth_try}.pkl'
with open(val_loss_path, 'rb') as f:
    val_loss_list = pickle.load(f)

val_non_weighted_loss_path = f'./DCNNoutput/{nth_try}/val_non_weighted_loss{nth_try}.pkl'
with open(val_non_weighted_loss_path, 'rb') as f:
    val_non_weighted_loss_list = pickle.load(f)


epoch_list = [i for i in range(0,n_epoch)]
# val_loss_list  = torch.tensor(val_loss_list, device = 'cpu')
# val_non_weighted_loss_list = torch.tensor(val_non_weighted_loss_list,device = 'cpu')
# train_loss_list  = torch.tensor(train_loss_list, device = 'cpu')
# train_non_weighted_loss_list = torch.tensor(train_non_weighted_loss_list,device = 'cpu')
plt.plot(epoch_list,val_loss_list,'tab:blue',label ='val weighted')
plt.plot(epoch_list,val_non_weighted_loss_list,'lightseagreen',label ='val non-weighted')
plt.plot(epoch_list,train_loss_list,'tab:red',label = 'train weighted')
plt.plot(epoch_list,train_non_weighted_loss_list,'lightsalmon',label = 'train non-weighted')
plt.ylim(0,3.5)
plt.xlim(0,500)
plt.title('Loss track plot')
plt.ylabel('Batch Loss')
plt.xlabel('Number of Epoch')
plt.legend(loc='upper left')
plt.savefig(loss_plot_name)
plt.close()

# plot accuracy changes
plt.plot(epoch_list,val_acc_list,'tab:blue',label ='val acc')
plt.plot(epoch_list,val_balanced_acc_list,'lightseagreen',label ='val balanced acc')
plt.plot(epoch_list,train_acc_list,'tab:red',label = 'train acc')
plt.plot(epoch_list,train_balanced_acc_list,'lightsalmon',label = 'train balanced acc')
plt.ylim(0.4,1)
plt.xlim(0,500)
plt.title('Accuracy track plot')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epoch')
plt.legend(loc='lower right')
plt.savefig(f'{acc_plot_name}')
plt.close()