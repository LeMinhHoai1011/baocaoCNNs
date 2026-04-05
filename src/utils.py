import matplotlib.pyplot as plt

def plot_metrics(results):
    for name,hist in results.items():
        plt.plot(hist["test_acc"],label=f"{name} Acc")
    plt.legend(); plt.title("Accuracy Comparison"); plt.show()

    for name,hist in results.items():
        plt.plot(hist["train_loss"],label=f"{name} Loss")
    plt.legend(); plt.title("Loss Comparison"); plt.show()

    for name,hist in results.items():
        print(f"{name} training time: {hist['time']:.2f} seconds")