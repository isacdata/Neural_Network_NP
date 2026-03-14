import numpy as np
import matplotlib.pyplot as plt
import itertools

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    return correct / total if total > 0 else 0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def classification_report(y_true, y_pred, show=False, location=None):
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if location:
        if show == True:
            plt.savefig(f"{location}/confusion_matrix.png", dpi=300)
            plt.show()
        else:
            plt.savefig(f"{location}/confusion_matrix.png", dpi=300)
            plt.close()
    else:
        if show == True:
            plt.savefig("confusion_matrix.png", dpi=300)
            plt.show()
        else:
            plt.savefig("confusion_matrix.png", dpi=300)
            plt.close()

    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

def erros_epocas(erros_historico, show=False, location=None):
    plt.figure(figsize=(10, 5))
    plt.plot(erros_historico, label='Erro Médio Quadrático')
    plt.title('Evolução do Erro ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if location:
        if show == True:
            plt.savefig(f"{location}/erros_epocas.png", dpi=300)
            plt.show()
        else:
            plt.savefig(f"{location}/erros_epocas.png", dpi=300)
            plt.close()
    else:
        if show == True:
            plt.savefig("erros_epocas.png", dpi=300)
            plt.show()
        else:
            plt.savefig("erros_epocas.png", dpi=300)
            plt.close()
