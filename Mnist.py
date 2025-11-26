import urllib.request
import gzip
import shutil
import os
import numpy as np



def download_mnist(save_dir="mnist_data"):
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"]

    os.makedirs(save_dir,exist_ok=True)
    for file in files:
        url = base_url + file
        save_path = os.path.join(save_dir,file)

        if os.path.exists(save_path):
            print(f"File {file} already exists. Skipping download.")
            continue
        print(f"Downloading {url}...")
        try:
            with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"Successfully downloaded and saved to {save_path}")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
            raise IOError(f"Failed to download {file}. Halting execution.") from e

def load_mnist_images(filename):
    with gzip.open(filename,'rb') as f:
        magic = np.frombuffer(f.read(4),dtype='>i4')
        if magic[0] != 2051:
            raise ValueError(f"Invalid magic number {magic[0]} in {filename}")
        num_images = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_rows = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_cols = np.frombuffer(f.read(4), dtype='>i4')[0]
        image_data = f.read()
        images = np.frombuffer(image_data,dtype=np.uint8)
        images = images.reshape(num_images, num_rows,num_cols)
        return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype='>i4')
        if magic[0] != 2049:
            raise ValueError(f"Invalid magic number {magic[0]} in {filename}")
        num_labels = np.frombuffer(f.read(4), dtype='>i4')[0]
        label_data = f.read()
        labels = np.frombuffer(label_data,dtype=np.uint8)
        return labels

def load_mnist(data_dir="mnist_data"):
    image_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    label_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    X_train = load_mnist_images(image_path)
    y_train = load_mnist_labels(label_path)

    image_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    label_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(image_path)
    y_test = load_mnist_labels(label_path)

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    download_mnist()
    print("Loading data into NumPy arrays...")

    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    print(f"\nTraining images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape:     {test_images.shape}")
    print(f"Test labels shape:     {test_labels.shape}")