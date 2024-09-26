import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys
# CUDAの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# オートエンコーダーの定義
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 変分オートエンコーダーの定義
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim * 2))  # 平均と標準偏差の2つ
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = h.chunk(2, dim=-1)
        return mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var


# 一クラスSVMのクラス
class OneClassSVMWrapper:
    def __init__(self):
        self.model = OneClassSVM(gamma='auto')

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


# Isolation Forestのクラス
class IsolationForestWrapper:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


# データセットの準備
def prepare_data(train_file, test_file, tokenizer_file):
    # 学習データの読み込み
    with open(train_file, 'r') as f:
        train_logs = f.readlines()

    # テストデータの読み込み
    with open(test_file, 'r') as f:
        test_logs = f.readlines()

    # テストデータのラベル作成
    test_texts = []
    test_labels = []
    for line in test_logs:
        parts = line.rsplit(' ', 1)
        test_texts.append(parts[0])
        if parts[1].strip() == '-':
            test_labels.append(0)
        else:
            test_labels.append(1)

    # 独自トークナイザーの準備
    tokenizer = BertWordPieceTokenizer(tokenizer_file)

    # トークン化とID列への変換
    train_encoded = tokenizer.encode_batch(train_logs)
    test_encoded = tokenizer.encode_batch(test_texts)

    train_input_ids = torch.tensor([enc.ids for enc in train_encoded])
    test_input_ids = torch.tensor([enc.ids for enc in test_encoded])

    max_len = max(train_input_ids.shape[1], test_input_ids.shape[1])

    # パディング
    train_input_ids = nn.functional.pad(train_input_ids, (0, max_len - train_input_ids.shape[1]), 'constant', 0)
    test_input_ids = nn.functional.pad(test_input_ids, (0, max_len - test_input_ids.shape[1]), 'constant', 0)

    return train_input_ids, test_input_ids, torch.tensor(test_labels)


# 学習と評価の関数
def train_autoencoder(model, dataloader, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for data in dataloader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, inputs.float())
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model


def test_autoencoder(model, dataloader, threshold=0.01):
    model.eval()
    anomalies = []
    true_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs.float())
            loss = nn.MSELoss()(outputs, inputs.float())
            anomalies.extend([1 if loss.item() > threshold else 0] * inputs.size(0))
            true_labels.extend(labels.tolist())

    return anomalies, true_labels


def train_vae(model, dataloader, num_epochs=50):
    def loss_function(recon_x, x, mean, log_var):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return BCE + KLD

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for data in dataloader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mean, log_var = model(inputs.float())
            loss = loss_function(recon_batch, inputs.float(), mean, log_var)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model


def test_vae(model, dataloader, threshold=0.01):
    model.eval()
    anomalies = []
    true_labels = []

    def loss_function(recon_x, x, mean, log_var):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return BCE + KLD

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            recon_batch, mean, log_var = model(inputs.float())
            loss = loss_function(recon_batch, inputs.float(), mean, log_var)
            anomalies.extend([1 if loss.item() > threshold else 0] * inputs.size(0))
            true_labels.extend(labels.tolist())

    return anomalies, true_labels


def train_sklearn_model(model, train_features):
    model.fit(train_features)
    return model


def test_sklearn_model(model, test_features):
    predictions = model.predict(test_features)
    anomalies = predictions == -1
    return anomalies


# メイン関数
def main(model_type, train_file, test_file, tokenizer_file, batch_size=32):
    train_file = f"datasets_for_models/sample_param/train/dataset_train_{'1000000'}.txt"
    test_file = f"datasets_for_models/sample_param/test/delete_MASK/dataset_test_{'state'}.txt"
    tokenizer_file = f'models/anomaly_detection/parameter/trained_tokenizer/vocab_size_{"10051"}/vocab.txt'
    train_input_ids, test_input_ids, test_labels = prepare_data(train_file, test_file, tokenizer_file)

    train_dataset = TensorDataset(train_input_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_input_ids, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if model_type == "autoencoder":
        input_dim = train_input_ids.size(1)
        hidden_dim = 256
        model = Autoencoder(input_dim, hidden_dim).to(device)
        model = train_autoencoder(model, train_dataloader)
        anomalies, true_labels = test_autoencoder(model, test_dataloader)
    elif model_type == "vae":
        input_dim = train_input_ids.size(1)
        hidden_dim = 256
        latent_dim = 64
        model = VAE(input_dim, hidden_dim, latent_dim).to(device)
        model = train_vae(model, train_dataloader)
        anomalies, true_labels = test_vae(model, test_dataloader)
    elif model_type == "oneclasssvm":
        train_features = train_input_ids.numpy()
        test_features = test_input_ids.numpy()
        model = OneClassSVMWrapper()
        model = train_sklearn_model(model, train_features)
        anomalies = test_sklearn_model(model, test_features)
        true_labels = test_labels.numpy()
    elif model_type == "isolationforest":
        train_features = train_input_ids.numpy()
        test_features = test_input_ids.numpy()
        model = IsolationForestWrapper()
        model = train_sklearn_model(model, train_features)
        anomalies = test_sklearn_model(model, test_features)
        true_labels = test_labels.numpy()
    else:
        raise ValueError("Invalid model type")

    # 精度の評価
    # print("Anomalies Detected:", anomalies)
    # print("Actual Labels:", true_labels)
    accuracy = accuracy_score(true_labels, anomalies)
    precision = precision_score(true_labels, anomalies, zero_division=1)
    recall = recall_score(true_labels, anomalies, zero_division=1)
    f1 = f1_score(true_labels, anomalies, zero_division=1)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    cm = confusion_matrix(true_labels, anomalies)
    tn, fp, fn, tp = cm.flatten()

    false_alarm_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
    underreport_rate = fn / (fn + tp) if (fn + tp) != 0 else 0
    eval_results = {
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "underreport_rate": underreport_rate,
        "rc": recall,
        "pc": precision,
        "acc": accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    result = f"{eval_results['f1']} {eval_results['false_alarm_rate']} {eval_results['underreport_rate']} {eval_results['rc']} {eval_results['pc']} {eval_results['acc']} " \
             f"{eval_results['tn']} {eval_results['fp']} {eval_results['fn']} {eval_results['tp']}"

    filename = "RESULTS_TEST_PCDS2024_Compatitive.txt"
    results_file_path = "results/" + filename
    command_line_string = " ".join(sys.argv)
    with open(results_file_path, "a+") as fw:
        fw.write(command_line_string + "\n")
        fw.write(result + "\n")

"""
python compatitive_unsupervised_main.py  --model_type autoencoder --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
python compatitive_unsupervised_main.py  --model_type vae --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
python compatitive_unsupervised_main.py  --model_type oneclasssvm --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
python compatitive_unsupervised_main.py  --model_type isolationforest --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["autoencoder", "vae", "oneclasssvm", "isolationforest"])
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the testing data file")
    parser.add_argument("--tokenizer_file", type=str, required=True, help="Path to the tokenizer file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    args = parser.parse_args()
    main(args.model_type, args.train_file, args.test_file, args.tokenizer_file, args.batch_size)
