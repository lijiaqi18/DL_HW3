import matplotlib.pyplot as plt

def plot_result_1(file_name:str, describe:str):
    loss_train = []
    acc_test = []
    f = open(file_name, "r")
    result = f.readlines()
    for res in result:
        if res[0:5] == 'Epoch':
            loss_train.append(float(res[18:24]))
            print(float(res[18:24]))
            acc_test.append(float(res[31:37]))
            print(float(res[31:37]))
        else:
            continue

    
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.plot(range(len(result)), loss_train)
    plt.savefig('./figure_result/train_loss_{}.png'.format(describe))
    plt.close()

    plt.title('Testing Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Test Acc')
    plt.plot(range(len(result)), acc_test)
    plt.savefig('./figure_result/acc_test_{}.png'.format(describe))
    plt.close()

def plot_result_2(file_name:str, describe:str):
    acc_train = []
    acc_valid = []
    acc_test = []
    f = open(file_name, "r")
    result = f.readlines()
    for res in result:
        if res[0:5] == 'Epoch':
            acc_train.append(float(res[19:25]))
            # print(float(res[18:24]))
            acc_valid.append(float(res[32:38]))
            acc_test.append(float(res[46:52]))
            # print(float(res[31:37]))
        else:
            continue

    
    plt.title('Training Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Train Acc')
    plt.plot(range(len(result)), acc_train)
    plt.savefig('./figure_result/train_acc_{}.png'.format(describe))
    plt.close()

    plt.title('Validing Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Valid Acc')
    plt.plot(range(len(result)), acc_valid)
    plt.savefig('./figure_result/valid_acc_{}.png'.format(describe))
    plt.close()

    plt.title('Testing Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Test Acc')
    plt.plot(range(len(result)), acc_test)
    plt.savefig('./figure_result/test_acc_{}.png'.format(describe))
    plt.close()

    plt.title('Compared Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(range(len(result)), acc_train)
    plt.plot(range(len(result)), acc_valid)
    plt.plot(range(len(result)), acc_test)
    plt.legend(['train', 'valid', 'test'], loc = 'lower right')
    plt.savefig('./figure_result/compared_acc_{}.png'.format(describe))
    plt.close()


plot_result_1("node2vec_run_result.txt", "node2vec")
plot_result_2("gcn_run_result.txt", "gcn")
plot_result_2("gat_run_result.txt", "gat")
plot_result_2("deepgcn_result_raw.txt", "deepgcn")
