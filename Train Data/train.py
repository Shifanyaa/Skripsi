import torch
import pandas as pd
import numpy as np
import argparse
import logging
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from models.dbn import DBN
from models.cnn import CNNModel
from models.random_forest import train_rf, evaluate_rf

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

def load_data(train_path, val_path, test_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Handle missing values (NaN) in target labels by dropping rows with NaN values in 'Label'
    train_df.dropna(subset=['Label'], inplace=True)
    val_df.dropna(subset=['Label'], inplace=True)
    test_df.dropna(subset=['Label'], inplace=True)

    # Convert the 'Label' column to binary (0 or 1)
    y_train = np.where(train_df['Label'] > 0, 1, 0)
    y_val = np.where(val_df['Label'] > 0, 1, 0)
    y_test = np.where(test_df['Label'] > 0, 1, 0)

    # Drop the 'Label' column and use other features for X
    X_train = train_df.drop(columns=['Label'])
    X_val = val_df.drop(columns=['Label'])
    X_test = test_df.drop(columns=['Label'])

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_torch_model(model, X, y):
    with torch.no_grad():
        preds = model(X).squeeze()
        pred_bin = (preds > 0.5).float()
        mcc = matthews_corrcoef(y.cpu().numpy(), pred_bin.cpu().numpy())
        auc = roc_auc_score(y.cpu().numpy(), preds.cpu().numpy())
        return mcc, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "dbn", "rf"], required=True)
    args = parser.parse_args()

    setup_logging()

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("train.csv", "val.csv", "test.csv")
    input_dim = X_train.shape[1]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    start_time = time.time()

    if args.model == "dbn":
        logging.info("Training DBN...")
        model = DBN(n_visible=input_dim, n_hidden=[128, 64])
        model.train_model(X_train_tensor, y_train_tensor, val_data=(X_val_tensor, y_val_tensor), epochs=50)
        mcc_val, auc_val = evaluate_torch_model(model, X_val_tensor, y_val_tensor)
        mcc_test, auc_test = evaluate_torch_model(model, X_test_tensor, y_test_tensor)

    elif args.model == "cnn":
        logging.info("Training CNN...")
        model = CNNModel(input_dim)
        model.train_model(X_train_tensor, y_train_tensor, val_data=(X_val_tensor, y_val_tensor), epochs=50)
        mcc_val, auc_val = evaluate_torch_model(model, X_val_tensor, y_val_tensor)
        mcc_test, auc_test = evaluate_torch_model(model, X_test_tensor, y_test_tensor)

    elif args.model == "rf":
        logging.info("Training Random Forest...")
        model = train_rf(X_train, y_train)
        mcc_val, auc_val = evaluate_rf(model, X_val, y_val)
        mcc_test, auc_test = evaluate_rf(model, X_test, y_test)

    end_time = time.time()
    logging.info(f"Training complete in {(end_time - start_time):.2f} seconds")
    logging.info(f"Validation MCC: {mcc_val:.4f}, AUC: {auc_val:.4f}")
    logging.info(f"Test MCC: {mcc_test:.4f}, AUC: {auc_test:.4f}")

if __name__ == "__main__":
    main()
