import os
import sys
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import json
from pathlib import Path
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score

# Tambahin root proyek ke sys.path
PROJECT_ROOT = "D:/athala-adjutor"
sys.path.append(PROJECT_ROOT)

try:
    from src.dataset.manager import DatasetManager
except ImportError as e:
    logging.error(f"Gagal impor DatasetManager: {e}")
    logging.error("Pastikan D:\\athala-adjutor\\src\\dataset\\manager.py ada dan __init__.py ada di src, src/ai, src/dataset")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AthalaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)

class AthalaClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AthalaClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Adjusted dropout rate
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)  # Adjusted dropout rate
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)  # Adjusted dropout rate
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

def load_processed_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text from various fields
        text = ''
        if 'text' in data:
            text = data['text']
        elif 'sample_content' in data:
            text = data['sample_content']
        elif 'sample_data' in data:
            if isinstance(data['sample_data'], list):
                text = ' '.join([str(item) for item in data['sample_data']])
            else:
                text = str(data['sample_data'])
        elif 'metadata_comment' in data:
            text = data['metadata_comment']
        elif 'metadata_description' in data:
            text = data['metadata_description']
        elif 'metadata_tool_techniques' in data:
            text = data['metadata_tool_techniques']
        elif 'metadata_tool_tactics' in data:
            text = data['metadata_tool_tactics']
        elif 'metadata_malwares_name' in data:
            text = data['metadata_malwares_name']
        elif 'metadata_groups_name' in data:
            text = data['metadata_groups_name']
        elif 'metadata_tags' in data:
            text = data['metadata_tags']
        elif 'dest_ip' in data:
            text = data['dest_ip']
        elif 'service_name' in data:
            text = data['service_name']
        elif 'domain' in data:
            text = data['domain']
        elif 'fuzzer' in data:
            text = data['fuzzer']
        elif 'signature' in data:
            text = data['signature']
        elif 'category' in data:
            text = data['category']
        elif 'ip' in data:
            text = data['ip']
        elif 'rows' in data:
            text = str(data['rows'])
        elif 'columns' in data:
            text = ' '.join(data['columns'])
        elif 'format' in data:
            text = data['format']
        elif 'size' in data:
            text = str(data['size'])
        elif 'mode' in data:
            text = data['mode']
        
        if not text:
            logger.warning(f"Tidak ada teks yang bisa diekstrak dari {file_path}. Struktur JSON: {str(data)[:200]}")
            return None, None, None
        
        # Extract label if available
        label = data.get('label', 
                        data.get('category',
                               data.get('metadata_category',
                                      data.get('type', 'unknown'))))
        
        return pd.Series([text]), pd.Series([label]), file_path.name
    except Exception as e:
        logger.error(f"Gagal load JSON {file_path}: {e}")
        return None, None, None

def load_processed_txt(file_path):
    """Load dan proses teks dari file TXT di data/processed.
    
    Args:
        file_path (Path): Path ke file TXT yang akan diproses
        
    Returns:
        tuple: (pd.Series text, pd.Series label, str filename) atau (None, None, None) jika gagal
    """
    try:
        # Baca file dengan encoding yang sesuai
        with open(file_path, "r", encoding='utf-8') as f:
            text = f.read()
            
        # Bersihkan teks
        text = text.strip()
        if not text:
            logger.warning(f"File TXT kosong: {file_path}")
            return None, None, None
            
        # Coba ekstrak label dari nama file
        filename = file_path.name
        if filename.startswith("extracted_"):
            # Untuk file hasil ekstrak PDF
            label = filename.replace("extracted_", "").split(".")[0]
        else:
            # Fallback ke parent directory
            label = file_path.parent.name
            
        # Validasi label
        if not label or label.lower() == "unknown":
            logger.warning(f"Label tidak valid untuk {file_path}")
            label = "unknown"
            
        return pd.Series([text]), pd.Series([label]), file_path.name
        
    except UnicodeDecodeError:
        logger.error(f"Gagal decode file {file_path}, mencoba dengan encoding lain")
        try:
            with open(file_path, "r", encoding='latin-1') as f:
                text = f.read()
            return pd.Series([text]), pd.Series([file_path.parent.name]), file_path.name
        except Exception as e:
            logger.error(f"Gagal total membaca file {file_path}: {e}")
            return None, None, None
    except Exception as e:
        logger.error(f"Gagal load TXT {file_path}: {e}")
        return None, None, None

def prepare_data(processed_dir, include_pdf=True):
    """Prepare data from processed files in data/processed."""
    processed_dir = Path(processed_dir)
    all_X = []
    all_y = []
    file_names = []
    skipped_files = []
    failed_files = []

    # Get all JSON files
    json_summary_files = list(processed_dir.glob("summary_*.json"))
    json_metadata_files = [f for f in processed_dir.glob("metadata_*.json") if not any(ext in f.name.lower() for ext in ['.jpg.json', '.png.json', '.jpeg.json'])]
    json_files = json_summary_files + json_metadata_files
    logger.info(f"Found {len(json_summary_files)} summary JSON files.")
    logger.info(f"Found {len(json_metadata_files)} relevant metadata JSON files.")
    logger.info(f"Total relevant JSON files found: {len(json_files)}")

    files_to_process = list(json_files) # Start with JSON files

    # Optional: include PDF text files
    if include_pdf:
        txt_files = list(processed_dir.glob("extracted_*.txt"))
        logger.info(f"Found {len(txt_files)} TXT files (extracted PDFs).")
        files_to_process.extend(txt_files)
    else:
         logger.info("Skipping TXT files as include_pdf is False.")

    total_files_found = len(files_to_process)
    logger.info(f"Total files found matching patterns (JSON + TXT): {total_files_found}")

    processed_count = 0
    attempted_files_count = 0
    for file_path in files_to_process:
        attempted_files_count += 1
        logger.debug(f"Attempting to process file {attempted_files_count}/{total_files_found}: {file_path.name}")
        X, y, fname = None, None, None # Reset for each file
        try:
            if file_path.suffix == '.json':
                X, y, fname = load_processed_json(file_path)
            elif file_path.suffix == '.txt':
                 X, y, fname = load_processed_txt(file_path)
            else:
                 logger.warning(f"Skipping file with unexpected suffix: {file_path}")
                 skipped_files.append({"file": str(file_path), "reason": "Unexpected suffix"})
                 continue

            if X is not None and y is not None and fname is not None:
                if not X.empty and not y.empty: # Ensure Series are not empty
                    all_X.append(X)
                    all_y.append(y)
                    file_names.append(fname)
                    processed_count += X.shape[0] # Count samples added
                    logger.debug(f"Successfully processed and added data from: {fname}")
                else:
                    logger.warning(f"Skipped file due to empty data/label Series: {file_path.name}")
                    skipped_files.append({"file": str(file_path), "reason": "Empty Series after load"})
            else:
                # load_processed_json/txt already logs warnings/errors for specific reasons (no text, load fail)
                # We'll count them as skipped here for summary, specific reason is in earlier logs
                logger.debug(f"Skipping file as loading returned None: {file_path.name}")
                skipped_files.append({"file": str(file_path), "reason": "Loading function returned None"})

        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {e}", exc_info=True) # Log stack trace
            failed_files.append({"file": str(file_path), "error": str(e)})

    logger.info(f"--- Processing Summary ---")
    logger.info(f"Total files found matching patterns: {total_files_found}")
    logger.info(f"Attempted to process: {attempted_files_count}")
    logger.info(f"Successfully loaded data from: {len(all_X)} files")
    logger.info(f"Total samples loaded: {processed_count}")
    logger.info(f"Files skipped (no data/label returned by loader): {len(skipped_files)}")
    logger.info(f"Files failed due to unexpected errors: {len(failed_files)}")
    # You could optionally log the lists of skipped/failed files if needed for debugging
    # logger.info(f"Skipped files list: {skipped_files}")
    # logger.info(f"Failed files list: {failed_files}")
    logger.info(f"--------------------------")

    if not all_X:
        logger.error("No data successfully loaded from data/processed.")
        return None, None, None, None

    # Combine all data
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    logger.info(f"Total data samples after concatenation: {len(X_combined)}")

    # --- Convert to Binary Classification ('unknown' vs 'known') ---
    original_class_counts = Counter(y_combined)
    logger.info(f"Original class distribution before binary conversion: {original_class_counts}")
    y_combined_binary = y_combined.apply(lambda label: 'unknown' if label == 'unknown' else 'known')
    binary_class_counts = Counter(y_combined_binary)
    logger.info(f"Binary class distribution ('unknown' vs 'known'): {binary_class_counts}")
    y_combined = y_combined_binary # Replace original labels with binary labels
    # --- End Binary Conversion ---

    # Check class distribution (should now be binary)
    class_counts = Counter(y_combined)
    logger.info(f"Class distribution BEFORE filtering: {class_counts}")

    # --- Filter out classes with only one sample --- 
    # This filtering might still be relevant if somehow 'known' or 'unknown' has only 1 sample (very unlikely)
    min_samples_required = 2
    classes_to_keep = {cls for cls, count in class_counts.items() if count >= min_samples_required}
    samples_to_keep_mask = y_combined.isin(classes_to_keep)

    if not samples_to_keep_mask.all(): # Check if any filtering is needed
        num_removed_classes = len(class_counts) - len(classes_to_keep)
        num_removed_samples = len(y_combined) - samples_to_keep_mask.sum()
        logger.warning(f"Removing {num_removed_classes} classes with less than {min_samples_required} samples. Total samples removed: {num_removed_samples}")
        X_combined = X_combined[samples_to_keep_mask]
        y_combined = y_combined[samples_to_keep_mask]
        # Reset index after filtering if needed, though usually not necessary for subsequent steps here
        X_combined = X_combined.reset_index(drop=True)
        y_combined = y_combined.reset_index(drop=True)
        logger.info(f"Total data samples AFTER filtering: {len(X_combined)}")
        logger.info(f"Class distribution AFTER filtering: {Counter(y_combined)}")
    else:
        logger.info("No classes needed filtering (all classes have >= {min_samples_required} samples).")
    # --- End filtering ---

    # Encode labels first to check number of classes
    label_encoder = LabelEncoder()
    try:
        y_encoded = label_encoder.fit_transform(y_combined)
        logger.info(f"Number of classes found: {len(label_encoder.classes_)}")
    except Exception as e:
        logger.error(f"Error encoding labels: {e}. Check label data.")
        return None, None, None, None

    # Handle case with only one class before proceeding
    if len(label_encoder.classes_) < 2:
        logger.warning("Only one class found in the data. Training cannot proceed with a single class.")
        return None, None, None, None # Stop processing if only one class

    # Encode features using TF-IDF without limiting features
    vectorizer = TfidfVectorizer()
    X_encoded = vectorizer.fit_transform(X_combined).toarray()
    logger.info(f"TF-IDF features: {X_encoded.shape}")

    # Return encoded data and encoders
    return X_encoded, y_encoded, label_encoder, vectorizer

def train_model(X, y, num_epochs=30):
    """Train the classification model."""
    # Define device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create dataset and dataloader
    train_dataset = AthalaDataset(X_train, y_train)
    test_dataset = AthalaDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize model
    num_classes = len(set(y))
    if num_classes < 2:
        logger.error("Training requires at least 2 classes, but found only {num_classes}. Aborting training.")
        return None, 0 # Cannot train with only one class
    model = AthalaClassifier(input_size=X.shape[1], hidden_size=256, num_classes=num_classes)
    model.to(device) # Move model to the defined device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device) # Move data to device
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] # Store probabilities for AUC calculations
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Store probabilities for AUC calculations
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)

    # Convert lists to numpy arrays for sklearn metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics AFTER the loop using accumulated results
    test_accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # AUC-ROC Calculation (Binary Case)
    auc_roc = "N/A"
    if num_classes == 2:
        try:
            # For binary classification, roc_auc_score expects probabilities of the positive class (assuming label 1 is positive)
            # Find which label ('known' or 'unknown') is encoded as 1 by the label encoder
            positive_class_label = 1 # Assume class 1 is the positive class for now
            auc_roc = roc_auc_score(all_labels, all_probs[:, positive_class_label])
        except ValueError as e:
            auc_roc = f"N/A (Error: {e})"
            logger.warning(f"Skipping AUC-ROC calculation: {e}")
    elif num_classes > 2: # Keep multi-class logic if needed later
        try:
            # Check if all classes are present in y_true
            if len(np.unique(all_labels)) == num_classes:
                 auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
            else:
                 auc_roc = "N/A (Not all classes present in test set)"
                 logger.warning(f"AUC-ROC not calculated: Only {len(np.unique(all_labels))} classes present in test set out of {num_classes}.")
        except ValueError as e:
            auc_roc = f"N/A (Error: {e})"
            logger.warning(f"Skipping AUC-ROC calculation: {e}")
    else:
        # num_classes < 2 case
        logger.warning("Skipping AUC-ROC calculation because there is only one class.")


    # Calculate AUC-PR for imbalanced data using accumulated probabilities
    auc_pr = 0.0
    if num_classes > 1:
        present_classes = np.unique(all_labels)
        for i in range(num_classes):
            if i in present_classes:
                # Ensure there are both positive and negative samples for the current class 'i'
                is_class_i = (all_labels == i).astype(int)
                if len(np.unique(is_class_i)) > 1:
                    p, r, _ = precision_recall_curve(is_class_i, all_probs[:, i])
                    # Handle cases where precision or recall might be empty or NaN
                    if len(r) > 0 and len(p) > 0:
                        # Integrate using AUC only if recall and precision arrays are valid
                        class_auc_pr = auc(r, p)
                        if not np.isnan(class_auc_pr):
                           auc_pr += class_auc_pr
                        else:
                            logger.warning(f"AUC-PR for class {i} resulted in NaN. Skipping.")
                    else:
                         logger.warning(f"Could not calculate Precision-Recall curve for class {i}. Skipping.")
                else:
                    logger.warning(f"Class {i} has only one type of sample (all positive or all negative) in test set. Skipping AUC-PR calculation for this class.")
            # No need for an else here, if class 'i' is not present, we don't calculate its AUC-PR

        # Average AUC-PR over the number of *present* classes to avoid division by zero if some classes aren't in the test set
        num_present_classes = len(present_classes)
        if num_present_classes > 0:
           auc_pr /= num_present_classes
        else:
           auc_pr = 0.0 # Or handle as N/A if no classes are present (edge case)
           logger.warning("No classes were present in the test set for AUC-PR calculation.")
    else:
        auc_pr = "N/A (Single class)"


    # Log all metrics
    logger.info(f"Test Accuracy: {test_accuracy*100:.2f}%")
    logger.info(f"Precision (Weighted): {precision:.4f}")
    logger.info(f"Recall (Weighted): {recall:.4f}")
    logger.info(f"F1-score (Weighted): {f1:.4f}")
    # Log AUC-ROC (handle string cases like N/A)
    logger.info(f"AUC-ROC (Binary/OVR Weighted): {auc_roc if isinstance(auc_roc, str) else f'{auc_roc:.4f}'}")
    # Log AUC-PR (handle string cases like N/A)
    logger.info(f"Test AUC-PR (Macro-averaged over present classes): {auc_pr if isinstance(auc_pr, str) else f'{auc_pr:.4f}'}")

    return model, test_accuracy

def cross_validate_model(X, y, num_epochs=20, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model, accuracy = train_model(X_train, y_train, num_epochs)
        accuracies.append(accuracy)
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    logger.info(f"Rata-rata akurasi cross-validation: {avg_accuracy:.2f}%")
    return model, avg_accuracy

def main(component="all"):
    """Jalankan pelatihan model."""
    os.chdir(PROJECT_ROOT)  # Set root proyek
    processed_dir = Path(PROJECT_ROOT) / "data" / "processed"

    if component == "all":
        logger.info(f"Memindai direktori terproses: {processed_dir}")
        if not processed_dir.exists():
            logger.error(f"Direktori {processed_dir} tidak ditemukan. Pastikan dataset sudah diproses.")
            return

        # Process all files without limits
        X, y, label_encoder, vectorizer = prepare_data(
            processed_dir,
            include_pdf=True  # Ensure TXT files are included
        )
        if X is None:
            logger.error("Failed to prepare data. Exiting.")
            return

        model, accuracy = train_model(X, y)
        if model is None:
             logger.error("Model training failed. Check logs for errors (e.g., insufficient classes).")
             return
             
        # Simpan model dan metadata
        model_path = Path(PROJECT_ROOT) / "models" / "athala_model.pth"
        vectorizer_path = Path(PROJECT_ROOT) / "models" / "vectorizer.pkl"
        label_encoder_path = Path(PROJECT_ROOT) / "models" / "label_encoder.pkl"
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        import joblib
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(label_encoder, label_encoder_path)
        logger.info(f"Model disimpan di {model_path}")
        logger.info(f"Vectorizer disimpan di {vectorizer_path}")
        logger.info(f"Label encoder disimpan di {label_encoder_path}")
    else:
        logger.info("Komponen tidak dikenal. Gunakan --component all")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Latih model untuk Athala-Adjutor")
    parser.add_argument("--component", default="all", choices=["all"],
                        help="Komponen untuk dijalankan: all")
    args = parser.parse_args()
    main(args.component)