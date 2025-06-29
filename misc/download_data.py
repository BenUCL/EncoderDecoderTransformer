import os
import pickle
import torchvision
import torchvision.transforms as transforms

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "MNIST", "raw")
TRAIN_FILENAME = "mnist_trainset.pkl"
TEST_FILENAME = "mnist_testset.pkl"

def main() -> None:
    """
    Download MNIST datasets using torchvision, then save them as pickles
    in the specified DATA_DIR.
    """
    # Ensure the output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Define the transform for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Download or load MNIST into ../data/MNIST
    base_dir = os.path.join(DATA_DIR, "..")
    trainset = torchvision.datasets.MNIST(
        root=base_dir,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.MNIST(
        root=base_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Save pickles in data/MNIST/raw
    train_path = os.path.join(DATA_DIR, TRAIN_FILENAME)
    with open(train_path, "wb") as f:
        pickle.dump(trainset, f)

    test_path = os.path.join(DATA_DIR, TEST_FILENAME)
    with open(test_path, "wb") as f:
        pickle.dump(testset, f)

    print(f"Saved trainset to {train_path}")
    print(f"Saved testset to {test_path}")

if __name__ == "__main__":
    main()
