import json
import matplotlib.pyplot as plt

# Load training history
history_path = "models/training_history.json"

try:
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["accuracy"], label="Training Accuracy", color="#10B981", linewidth=2)
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", color="#3B82F6", linewidth=2)
    plt.title("Model Accuracy Convergence", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss"], label="Training Loss", color="#EF4444", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color="#F59E0B", linewidth=2)
    plt.title("Model Loss Convergence", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)

    plt.tight_layout()
    plot_save_path = "models/training_curves.png"
    plt.savefig(plot_save_path, dpi=300)
    print(f"📊 Convergence curves plotted and saved successfully to: {plot_save_path}")
    plt.show()

except FileNotFoundError:
    print(f"Error: '{history_path}' not found. Make sure you placed your downloaded training_history.json inside the 'models' folder.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
