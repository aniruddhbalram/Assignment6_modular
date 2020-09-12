import matplotlib.pyplot as plt
import numpy as np

#print(test_accuracies.keys())



def plot_model_comparison(legend_list, model1_acc_hist, model1_loss_hist,
                          model2_acc_hist, model2_loss_hist,
                          model3_acc_hist, model3_loss_hist,
                          model4_acc_hist, model4_loss_hist,
                          model5_acc_hist, model5_loss_hist):
    fig, axs = plt.subplots(1,2,figsize=(20,5))
    # summarize history for accuracy
    x_size = len(model1_acc_hist)-1

    axs[0].plot(range(1,x_size+1), model1_acc_hist[1:])
    axs[0].plot(range(1,x_size+1), model2_acc_hist[1:])
    axs[0].plot(range(1,x_size+1), model3_acc_hist[1:])
    axs[0].plot(range(1,x_size+1), model4_acc_hist[1:])
    axs[0].plot(range(1,x_size+1), model5_acc_hist[1:])

    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,x_size+1),x_size/10)
    axs[0].legend(legend_list, loc='best')

   # plot losses
    axs[1].plot(range(1,x_size+1),model1_loss_hist[1:])
    axs[1].plot(range(1,x_size+1),model2_loss_hist[1:])
    axs[1].plot(range(1,x_size+1),model3_loss_hist[1:])
    axs[1].plot(range(1,x_size+1),model4_loss_hist[1:])
    axs[1].plot(range(1,x_size+1),model5_loss_hist[1:])
    axs[1].set_title('Model Losses')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,x_size+1),x_size/10)
    axs[1].legend(legend_list, loc='best')
    plt.show()
    fig.savefig("model_compare.png")
