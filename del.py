# how to run
# python3 ml-classifier.py --test-files flows_with-additional-insights/pfcp_5min_flows.csv flows_with-additional-insights/pfcp_10min_flows.csv flows_with-additional-insights/nidd_5min_flows.csv flows_with-additional-insights/nidd_10min_flows.csv --train-files  flows_with-additional-insights/pfcp_5min_flows.csv flows_with-additional-insights/pfcp_10min_flows.csv flows_with-additional-insights/nidd_5min_flows.csv flows_with-additional-insights/nidd_10min_flows.csv   --svm-scaler minmax --rf-scaler standard --knn-scaler minmax --ensemble-method random

import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import random


def save_classification_report(report, model_name, output_file="classification_reports.csv"):

    # Convert the classification report dictionary to a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Add model name column
    report_df['model'] = model_name

    # Save to CSV, append if file exists
    report_df.to_csv(output_file,
                     mode='a',
                     header=not pd.io.common.file_exists(output_file))

    print(f"{model_name} classification report saved to {output_file}")

# Preprocessing data
def preprocess_data(file_paths):
    datasets = []
    for file in file_paths:
        df = pd.read_csv(file, index_col=0)
        df.reset_index(inplace=True)
        print(df.head())

        # Convert IAT dictionaries if present
        for col in ['incoming_iats', 'outgoing_iats']:
            if col in df.columns:
                df[f'{col}_max'] = df[col].apply(lambda x: max(eval(x)) if x != "[]" else 0)
                df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(eval(x)) if x != "[]" else 0)
                df[f'{col}_std'] = df[col].apply(lambda x: np.std(eval(x)) if x != "[]" else 0)
                df[f'{col}_q3'] = df[col].apply(lambda x: np.percentile(eval(x), 75) if x != "[]" else 0)
                df = df.drop(columns=[col])

        datasets.append(df)

    combined_data = pd.concat(datasets, ignore_index=True)

    # Convert label to binary classes: 'B' -> 0 (benign), 'M' -> 1 (malicious)
    y = combined_data['label'].map({'B': 0, 'M': 1}).astype(int)
    X = combined_data.drop(columns=['label'])

    return X, y


# Scaling data
def scale_data(X_train, X_test, scaler_type):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaler type. Use 'standard' or 'minmax'.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def one_class_svm(X_train, X_test):

    clf = OneClassSVM(kernel='rbf', nu=0.1)
    clf.fit(X_train)
    # Convert predictions to binary format (-1 -> 0, 1 -> 1)
    y_pred_test = (clf.predict(X_test) == -1).astype(int)
    y_score_test = clf.score_samples(X_test)
    return y_pred_test, y_score_test

def elliptic_envelope(X_train, X_test, contamination=0.1):

    clf = EllipticEnvelope(contamination=contamination, random_state=42)
    clf.fit(X_train)
    # Convert predictions to binary format (-1 -> 0, 1 -> 1)
    y_pred_test = (clf.predict(X_test) == -1).astype(int)
    y_score_test = clf.score_samples(X_test)
    return y_pred_test, y_score_test

def local_outlier_factor(X_train, X_test, n_neighbors=20):

    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    clf.fit(X_train)
    # Convert predictions to binary format (-1 -> 0, 1 -> 1)
    y_pred_test = (clf.predict(X_test) == -1).astype(int)
    y_score_test = clf.score_samples(X_test)
    return y_pred_test, y_score_test
def run_one_class_classifiers(X_train, X_test, y_test, scalers):
    results = {}

    # One-class SVM
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['svm'])
    y_pred_ocsvm, y_score_ocsvm = one_class_svm(X_train_scaled, X_test_scaled)
    print("\nOne-class SVM Classification Report:")
    report_ocsvm = classification_report(y_test, y_pred_ocsvm, output_dict=True)
    save_classification_report(report_ocsvm, "One-class SVM")
    results['one_class_svm'] = (y_pred_ocsvm, y_score_ocsvm)

    # Elliptic Envelope
    y_pred_ee, y_score_ee = elliptic_envelope(X_train_scaled, X_test_scaled)
    print("\nElliptic Envelope Classification Report:")
    report_ee = classification_report(y_test, y_pred_ee, output_dict=True)
    save_classification_report(report_ee, "Elliptic Envelope")
    results['elliptic_envelope'] = (y_pred_ee, y_score_ee)

    # Local Outlier Factor
    y_pred_lof, y_score_lof = local_outlier_factor(X_train_scaled, X_test_scaled)
    print("\nLocal Outlier Factor Classification Report:")
    report_lof = classification_report(y_test, y_pred_lof, output_dict=True)
    save_classification_report(report_lof, "Local Outlier Factor")
    results['local_outlier_factor'] = (y_pred_lof, y_score_lof)

    return results


# Multi-class SVM classifier
def svm_multiclass(X_train, y_train, X_test):

    clf = SVC(probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return y_pred, y_prob


# Random Forest classifier
def random_forest_multiclass(X_train, y_train, X_test):

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return y_pred, y_prob


# K-Nearest Neighbors classifier
def knn_multiclass(X_train, y_train, X_test):

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return y_pred, y_prob


# Ensemble voting classifier
def ensemble_voting(X_train, y_train, X_test, ensemble_method):

    estimators = [
        ('svm', SVC(probability=True, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('knn', KNeighborsClassifier())
    ]

    if ensemble_method == 'highest_confidence':
        clf = VotingClassifier(estimators=estimators, voting='soft')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    elif ensemble_method == 'p1-p2':
        # Train individual classifiers
        clfs = {}
        probas = {}
        for name, clf in estimators:
            clfs[name] = clf.fit(X_train, y_train)
            probas[name] = clf.predict_proba(X_test)

        # Calculate P1-P2 differences for each classifier
        diffs = {}
        for name in clfs.keys():
            diffs[name] = np.abs(probas[name][:, 1] - probas[name][:, 0])

        # Select classifier with highest P1-P2 difference for each sample
        y_pred = np.zeros(len(X_test))
        for i in range(len(X_test)):
            best_clf = max(diffs.keys(), key=lambda k: diffs[k][i])
            y_pred[i] = clfs[best_clf].predict(X_test[i].reshape(1, -1))

    elif ensemble_method == 'random':
        # Train all classifiers
        predictions = {}
        for name, clf in estimators:
            clf.fit(X_train, y_train)
            predictions[name] = clf.predict(X_test)

        # Randomly select predictions
        y_pred = np.zeros(len(X_test))
        for i in range(len(X_test)):
            selected_clf = random.choice(list(predictions.keys()))
            y_pred[i] = predictions[selected_clf][i]

    else:
        raise ValueError("Unsupported ensemble method.")

    return y_pred

def run_multi_class_classifiers(X_train, X_test, y_train, y_test, scalers, ensemble_method):
    results = {}

    # SVM classification
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['svm'])
    y_pred_svm, y_prob_svm = svm_multiclass(X_train_scaled, y_train, X_test_scaled)
    print("\nSVM Classification Report:")
    report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
    save_classification_report(report_svm, "SVM")
    results['svm'] = (y_pred_svm, y_prob_svm)

    # Random Forest classification
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['rf'])
    y_pred_rf, y_prob_rf = random_forest_multiclass(X_train_scaled, y_train, X_test_scaled)
    print("\nRandom Forest Classification Report:")
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    save_classification_report(report_rf, "Random Forest")
    results['random_forest'] = (y_pred_rf, y_prob_rf)

    # KNN classification
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['knn'])
    y_pred_knn, y_prob_knn = knn_multiclass(X_train_scaled, y_train, X_test_scaled)
    print("\nKNN Classification Report:")
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    save_classification_report(report_knn, "KNN")
    results['knn'] = (y_pred_knn, y_prob_knn)

    # Ensemble classification with specified method
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, 'standard')  # Using standard scaling for ensemble
    y_pred_ensemble = ensemble_voting(X_train_scaled, y_train, X_test_scaled, ensemble_method)
    print(f"\nEnsemble Classification Report (Method: {ensemble_method}):")
    report_ensemble = classification_report(y_test, y_pred_ensemble, output_dict=True)
    save_classification_report(report_ensemble, f"Ensemble_{ensemble_method}")
    results['ensemble'] = y_pred_ensemble

    return results


def main():
    parser = argparse.ArgumentParser(description="Network Traffic Classification")
    parser.add_argument('--train-files', nargs='+', required=True, help='Paths to training CSV files')
    parser.add_argument('--test-files', nargs='+', required=True, help='Paths to testing CSV files')
    parser.add_argument('--svm-scaler', choices=['standard', 'minmax'], default='standard', help='Scaler for SVM')
    parser.add_argument('--rf-scaler', choices=['standard', 'minmax'], default='minmax',
                        help='Scaler for Random Forest')
    parser.add_argument('--knn-scaler', choices=['standard', 'minmax'], default='minmax', help='Scaler for KNN')
    parser.add_argument('--ensemble-method', choices=['highest_confidence', 'p1-p2', 'random'],
                        default='highest_confidence', help='Ensemble method for voting classifier')
    args = parser.parse_args()

    # Preprocess data
    print("\nProcessing training data...")
    X_train, y_train = preprocess_data(args.train_files)
    print("\nProcessing testing data...")
    X_test, y_test = preprocess_data(args.test_files)

    # Define scalers for different algorithms
    scalers = {
        'svm': args.svm_scaler,
        'rf': args.rf_scaler,
        'knn': args.knn_scaler
    }

    print("\n" + "=" * 50)
    print("Running One-Class Classifiers")
    print("=" * 50)
    one_class_results = run_one_class_classifiers(X_train, X_test, y_test, scalers)

    print("\n" + "=" * 50)
    print("Running Multi-Class Classifiers")
    print("=" * 50)
    multi_class_results = run_multi_class_classifiers(X_train, X_test, y_train, y_test,
                                                      scalers, args.ensemble_method)

    print("\nAll results have been saved to classification_reports.csv")
    print(f"Ensemble method used for multi-class classification: {args.ensemble_method}")


if __name__ == '__main__':
    main()


#how to run
#python3 ml-classifier.py --train-files folds/scenario_b/train_fold_1.csv  --test-files  folds/scenario_b/test_fold_1.csv --ensemble-method highest_confidence --svm-scaler standard --rf-scaler minmax --knn-scaler minmax



# import argparse
# import pandas as pd
# import numpy as np
# from sklearn.svm import SVC, OneClassSVM
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
# from sklearn.covariance import EllipticEnvelope
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import classification_report
# from sklearn.ensemble import VotingClassifier
# import random
#
#
# def save_classification_report(report, model_name, output_file="classification_reports.csv"):
#
#     # Convert the classification report dictionary to a DataFrame
#     report_df = pd.DataFrame(report).transpose()
#
#     # Add model name column
#     report_df['model'] = model_name
#
#     # Save to CSV, append if file exists
#     report_df.to_csv(output_file,
#                      mode='a',
#                      header=not pd.io.common.file_exists(output_file))
#
#     print(f"{model_name} classification report saved to {output_file}")
#
#
# def ip_to_int(ip):
#     """Convert IP address to integer, handling NaN and invalid values"""
#     if pd.isna(ip):
#         return 0
#     try:
#         return int(''.join([f"{int(i):03d}" for i in str(ip).split('.')]))
#     except:
#         return 0
#
#
# def preprocess_data(file_paths):
#     datasets = []
#     for file in file_paths:
#         df = pd.read_csv(file)
#
#         # Explicitly drop the flags column if it exists
#         if 'flags' in df.columns:
#             df = df.drop('flags', axis=1)
#
#         # Convert timestamp to datetime and extract features
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         df['hour'] = df['timestamp'].dt.hour
#         df['minute'] = df['timestamp'].dt.minute
#         df['second'] = df['timestamp'].dt.second
#
#         # Convert protocol to numeric using label encoding
#         df['protocol'] = pd.Categorical(df['protocol']).codes
#
#         # Convert IP addresses to numeric representations with error handling
#         df['src_ip'] = df['src_ip'].apply(ip_to_int)
#         df['dst_ip'] = df['dst_ip'].apply(ip_to_int)
#
#         # Handle ports - fill NaN with -1
#         df['src_port'] = df['src_port'].fillna(-1).astype(int)
#         df['dst_port'] = df['dst_port'].fillna(-1).astype(int)
#
#         # Handle packet_length - fill NaN with 0
#         df['packet_length'] = df['packet_length'].fillna(0).astype(int)
#
#         # Convert attack_type to numeric using label encoding
#         if 'attack_type' in df.columns:
#             df['attack_type'] = pd.Categorical(df['attack_type']).codes
#
#         # Drop timestamp column as we've extracted features from it
#         df.drop('timestamp', axis=1, inplace=True)
#
#         # Print the columns to verify
#         print("Columns after preprocessing:", df.columns.tolist())
#
#         datasets.append(df)
#
#     combined_data = pd.concat(datasets, ignore_index=True)
#
#     # Convert label to binary classes: 'Benign' -> 0, everything else -> 1
#     y = (combined_data['Label'] != 'Benign').astype(int)
#
#     # Drop the Label column
#     X = combined_data.drop(columns=['Label'])
#
#     return X.fillna(0), y
#
# # Scaling data
# def scale_data(X_train, X_test, scaler_type):
#     if scaler_type == 'standard':
#         scaler = StandardScaler()
#     elif scaler_type == 'minmax':
#         scaler = MinMaxScaler()
#     else:
#         raise ValueError("Unsupported scaler type. Use 'standard' or 'minmax'.")
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     return X_train_scaled, X_test_scaled
#
# def one_class_svm(X_train, X_test):
#     """
#     Implements one-class SVM classification
#     """
#     clf = OneClassSVM(kernel='rbf', nu=0.1)
#     clf.fit(X_train)
#     # Convert predictions to binary format (-1 -> 0, 1 -> 1)
#     y_pred_test = (clf.predict(X_test) == -1).astype(int)
#     y_score_test = clf.score_samples(X_test)
#     return y_pred_test, y_score_test
#
# def elliptic_envelope(X_train, X_test, contamination=0.1):
#     """
#     Implements Elliptic Envelope classification
#     """
#     clf = EllipticEnvelope(contamination=contamination, random_state=42)
#     clf.fit(X_train)
#     # Convert predictions to binary format (-1 -> 0, 1 -> 1)
#     y_pred_test = (clf.predict(X_test) == -1).astype(int)
#     y_score_test = clf.score_samples(X_test)
#     return y_pred_test, y_score_test
#
# def local_outlier_factor(X_train, X_test, n_neighbors=20):
#     """
#     Implements Local Outlier Factor classification
#     """
#     clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
#     clf.fit(X_train)
#     # Convert predictions to binary format (-1 -> 0, 1 -> 1)
#     y_pred_test = (clf.predict(X_test) == -1).astype(int)
#     y_score_test = clf.score_samples(X_test)
#     return y_pred_test, y_score_test
# def run_one_class_classifiers(X_train, X_test, y_test, scalers):
#     results = {}
#
#     # One-class SVM
#     X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['svm'])
#     y_pred_ocsvm, y_score_ocsvm = one_class_svm(X_train_scaled, X_test_scaled)
#     print("\nOne-class SVM Classification Report:")
#     report_ocsvm = classification_report(y_test, y_pred_ocsvm, output_dict=True)
#     save_classification_report(report_ocsvm, "One-class SVM")
#     results['one_class_svm'] = (y_pred_ocsvm, y_score_ocsvm)
#
#     # Elliptic Envelope
#     y_pred_ee, y_score_ee = elliptic_envelope(X_train_scaled, X_test_scaled)
#     print("\nElliptic Envelope Classification Report:")
#     report_ee = classification_report(y_test, y_pred_ee, output_dict=True)
#     save_classification_report(report_ee, "Elliptic Envelope")
#     results['elliptic_envelope'] = (y_pred_ee, y_score_ee)
#
#     # Local Outlier Factor
#     y_pred_lof, y_score_lof = local_outlier_factor(X_train_scaled, X_test_scaled)
#     print("\nLocal Outlier Factor Classification Report:")
#     report_lof = classification_report(y_test, y_pred_lof, output_dict=True)
#     save_classification_report(report_lof, "Local Outlier Factor")
#     results['local_outlier_factor'] = (y_pred_lof, y_score_lof)
#
#     return results
#
#
# # Multi-class SVM classifier
# def svm_multiclass(X_train, y_train, X_test):
#     """
#     Implements multi-class SVM classification
#
#     Parameters:
#     X_train: Training features
#     y_train: Training labels
#     X_test: Test features
#
#     Returns:
#     tuple: (predictions, probability scores)
#     """
#     clf = SVC(probability=True, random_state=42)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     y_prob = clf.predict_proba(X_test)
#     return y_pred, y_prob
#
#
# # Random Forest classifier
# def random_forest_multiclass(X_train, y_train, X_test):
#     """
#     Implements Random Forest classification
#
#     Parameters:
#     X_train: Training features
#     y_train: Training labels
#     X_test: Test features
#
#     Returns:
#     tuple: (predictions, probability scores)
#     """
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     y_prob = clf.predict_proba(X_test)
#     return y_pred, y_prob
#
#
# # K-Nearest Neighbors classifier
# def knn_multiclass(X_train, y_train, X_test):
#     """
#     Implements k-Nearest Neighbors classification
#
#     Parameters:
#     X_train: Training features
#     y_train: Training labels
#     X_test: Test features
#
#     Returns:
#     tuple: (predictions, probability scores)
#     """
#     clf = KNeighborsClassifier()
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     y_prob = clf.predict_proba(X_test)
#     return y_pred, y_prob
#
#
# # Ensemble voting classifier
# def ensemble_voting(X_train, y_train, X_test, ensemble_method):
#     """
#     Implements ensemble voting with multiple methods
#
#     Parameters:
#     X_train: Training features
#     y_train: Training labels
#     X_test: Test features
#     ensemble_method: Method for combining predictions ('highest_confidence', 'p1-p2', 'random')
#
#     Returns:
#     array: Final predictions
#     """
#     estimators = [
#         ('svm', SVC(probability=True, random_state=42)),
#         ('rf', RandomForestClassifier(random_state=42)),
#         ('knn', KNeighborsClassifier())
#     ]
#
#     if ensemble_method == 'highest_confidence':
#         clf = VotingClassifier(estimators=estimators, voting='soft')
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#
#     elif ensemble_method == 'p1-p2':
#         # Train individual classifiers
#         clfs = {}
#         probas = {}
#         for name, clf in estimators:
#             clfs[name] = clf.fit(X_train, y_train)
#             probas[name] = clf.predict_proba(X_test)
#
#         # Calculate P1-P2 differences for each classifier
#         diffs = {}
#         for name in clfs.keys():
#             diffs[name] = np.abs(probas[name][:, 1] - probas[name][:, 0])
#
#         # Select classifier with highest P1-P2 difference for each sample
#         y_pred = np.zeros(len(X_test))
#         for i in range(len(X_test)):
#             best_clf = max(diffs.keys(), key=lambda k: diffs[k][i])
#             y_pred[i] = clfs[best_clf].predict(X_test[i].reshape(1, -1))
#
#     elif ensemble_method == 'random':
#         # Train all classifiers
#         predictions = {}
#         for name, clf in estimators:
#             clf.fit(X_train, y_train)
#             predictions[name] = clf.predict(X_test)
#
#         # Randomly select predictions
#         y_pred = np.zeros(len(X_test))
#         for i in range(len(X_test)):
#             selected_clf = random.choice(list(predictions.keys()))
#             y_pred[i] = predictions[selected_clf][i]
#
#     else:
#         raise ValueError("Unsupported ensemble method.")
#
#     return y_pred
#
# def run_multi_class_classifiers(X_train, X_test, y_train, y_test, scalers, ensemble_method):
#     results = {}
#
#     # SVM classification
#     X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['svm'])
#     y_pred_svm, y_prob_svm = svm_multiclass(X_train_scaled, y_train, X_test_scaled)
#     print("\nSVM Classification Report:")
#     report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
#     save_classification_report(report_svm, "SVM")
#     results['svm'] = (y_pred_svm, y_prob_svm)
#
#     # Random Forest classification
#     X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['rf'])
#     y_pred_rf, y_prob_rf = random_forest_multiclass(X_train_scaled, y_train, X_test_scaled)
#     print("\nRandom Forest Classification Report:")
#     report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
#     save_classification_report(report_rf, "Random Forest")
#     results['random_forest'] = (y_pred_rf, y_prob_rf)
#
#     # KNN classification
#     X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scalers['knn'])
#     y_pred_knn, y_prob_knn = knn_multiclass(X_train_scaled, y_train, X_test_scaled)
#     print("\nKNN Classification Report:")
#     report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
#     save_classification_report(report_knn, "KNN")
#     results['knn'] = (y_pred_knn, y_prob_knn)
#
#     # Ensemble classification with specified method
#     X_train_scaled, X_test_scaled = scale_data(X_train, X_test, 'standard')  # Using standard scaling for ensemble
#     y_pred_ensemble = ensemble_voting(X_train_scaled, y_train, X_test_scaled, ensemble_method)
#     print(f"\nEnsemble Classification Report (Method: {ensemble_method}):")
#     report_ensemble = classification_report(y_test, y_pred_ensemble, output_dict=True)
#     save_classification_report(report_ensemble, f"Ensemble_{ensemble_method}")
#     results['ensemble'] = y_pred_ensemble
#
#     return results
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Network Traffic Classification")
#     parser.add_argument('--train-files', nargs='+', required=True, help='Paths to training CSV files')
#     parser.add_argument('--test-files', nargs='+', required=True, help='Paths to testing CSV files')
#     parser.add_argument('--svm-scaler', choices=['standard', 'minmax'], default='standard', help='Scaler for SVM')
#     parser.add_argument('--rf-scaler', choices=['standard', 'minmax'], default='minmax',
#                         help='Scaler for Random Forest')
#     parser.add_argument('--knn-scaler', choices=['standard', 'minmax'], default='minmax', help='Scaler for KNN')
#     parser.add_argument('--ensemble-method', choices=['highest_confidence', 'p1-p2', 'random'],
#                         default='highest_confidence', help='Ensemble method for voting classifier')
#     args = parser.parse_args()
#     # Preprocess data
#     print("\nProcessing training data...")
#     X_train, y_train = preprocess_data(args.train_files)
#     print("\nProcessing testing data...")
#     X_test, y_test = preprocess_data(args.test_files)
#
#     # Define scalers for different algorithms
#     scalers = {
#         'svm': args.svm_scaler,
#         'rf': args.rf_scaler,
#         'knn': args.knn_scaler
#     }
#
#     print("\n" + "=" * 50)
#     print("Running One-Class Classifiers")
#     print("=" * 50)
#     one_class_results = run_one_class_classifiers(X_train, X_test, y_test, scalers)
#
#     print("\n" + "=" * 50)
#     print("Running Multi-Class Classifiers")
#     print("=" * 50)
#     multi_class_results = run_multi_class_classifiers(X_train, X_test, y_train, y_test,
#                                                       scalers, args.ensemble_method)
#
#     print("\nAll results have been saved to classification_reports.csv")
#     print(f"Ensemble method used for multi-class classification: {args.ensemble_method}")
#
#
# if __name__ == '__main__':
#     main()




import boto3
import json
import random
import os
from botocore.exceptions import ClientError

AMI_ID = ''        
SECURITY_GROUP_ID = '' 
IAM_ROLE = ''         
VPC_ID = ''       
KEY_NAME = '-arm'
STORAGE_SIZE = 150

# Define fallback configurations
INSTANCE_TYPE_FALLBACKS = {
    't4g.2xlarge': ['t4g.xlarge', 't4g.large'],
    't3.2xlarge': ['t3.xlarge', 't3.large']
}

ec2_client = boto3.client('ec2')

def parse_body(event):
    if 'body' in event:
        return json.loads(event['body'])
    return {}

def get_availability_zones():
    response = ec2_client.describe_availability_zones(
        Filters=[{'Name': 'region-name', 'Values': [ec2_client.meta.region_name]}]
    )
    return [az['ZoneName'] for az in response['AvailabilityZones']]

def get_subnet_for_az(az):
    subnets = ec2_client.describe_subnets(
        Filters=[
            {'Name': 'vpc-id', 'Values': [VPC_ID]},
            {'Name': 'availability-zone', 'Values': [az]}
        ]
    )['Subnets']
    return random.choice(subnets)['SubnetId'] if subnets else None

def lambda_handler(event, context):
    token = event['headers'].get('Authorization')
    secret = os.environ.get('header')
    
    if not (token and token == secret):
        return {
            'statusCode': 401,
            'body': json.dumps('401 Unauthorized')
        }
    
    try:
        body = parse_body(event)
        
        if event['resource'] == '/provision-machine':
            print('create ec2 api was called...')
            
            arch = body.get('arch')
            if arch == "graviton":
                instance_type = body.get('instance_type', 't4g.2xlarge')
                AMI_ID = "ami-xxx"
            else:
                instance_type = body.get('instance_type', 't3.2xlarge')
                AMI_ID = "ami-xxxx"

            is_spot = body.get('spot', True)  # Default to spot instance
            
            # Get the first AZ's subnet to start with
            availability_zones = get_availability_zones()
            subnet_id = get_subnet_for_az(availability_zones[0])
            
            if is_spot:
                launch_spec = {
                    'ImageId': AMI_ID,
                    'InstanceType': instance_type,
                    'KeyName': KEY_NAME,
                    'SecurityGroupIds': [SECURITY_GROUP_ID],
                    'IamInstanceProfile': {'Name': IAM_ROLE},
                    'SubnetId': subnet_id
                }
                
                # Create the Spot Instance request
                spot_response = ec2_client.request_spot_instances(
                    InstanceCount=1,
                    Type='one-time',
                    LaunchSpecification=launch_spec
                )
                
                request_id = spot_response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                return {
                    'statusCode': 202,
                    'body': json.dumps({
                        'RequestId': request_id,
                        'Status': 'pending',
                        'InitialInstanceType': instance_type,
                        'IsSpot': True
                    })
                }
            else:
                # For on-demand, try immediate launch
                instance_params = {
                    'ImageId': AMI_ID,
                    'InstanceType': instance_type,
                    'KeyName': KEY_NAME,
                    'SecurityGroupIds': [SECURITY_GROUP_ID],
                    'SubnetId': subnet_id,
                    'MaxCount': 1,
                    'MinCount': 1,
                    'IamInstanceProfile': {'Name': IAM_ROLE}
                }
                
                try:
                    response = ec2_client.run_instances(**instance_params)
                    instance = response['Instances'][0]
                    return {
                        'statusCode': 202,
                        'body': json.dumps({
                            'InstanceId': instance['InstanceId'],
                            'Status': 'pending',
                            'IsSpot': False
                        })
                    }
                except ClientError as e:
                    if e.response['Error']['Code'] in ['InsufficientInstanceCapacity', 'InsufficientHostCapacity']:
                        # If immediate launch fails, fall back to spot request
                        spot_response = ec2_client.request_spot_instances(
                            InstanceCount=1,
                            Type='one-time',
                            LaunchSpecification=launch_spec
                        )
                        request_id = spot_response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                        return {
                            'statusCode': 202,
                            'body': json.dumps({
                                'RequestId': request_id,
                                'Status': 'pending',
                                'InitialInstanceType': instance_type,
                                'IsSpot': True,
                                'FallbackFromOnDemand': True
                            })
                        }
                    raise

        elif event['resource'] == '/checkstatus':
            print('check status api was called...')
            
            body = parse_body(event)
            request_id = body.get('request_id')
            
            spot_result = ec2_client.describe_spot_instance_requests(SpotInstanceRequestIds=[request_id])
            spot_request = spot_result['SpotInstanceRequests'][0]
            
            if spot_request['State'] == 'failed':
                # If the current attempt failed, try with next instance type or AZ
                original_error = spot_request.get('Status', {}).get('Message', '')
                
                # Get the current instance type from the launch specification
                current_type = spot_request['LaunchSpecification']['InstanceType']
                current_az = spot_request['LaunchSpecification']['SubnetId']
                
                # Try to find next configuration
                fallback_types = INSTANCE_TYPE_FALLBACKS.get(current_type, [])
                availability_zones = get_availability_zones()
                
                # Try next AZ first
                current_az_index = next((i for i, az in enumerate(availability_zones) 
                                      if get_subnet_for_az(az) == current_az), -1)
                if current_az_index < len(availability_zones) - 1:
                    # Try next AZ with same instance type
                    next_az = availability_zones[current_az_index + 1]
                    next_subnet = get_subnet_for_az(next_az)
                elif fallback_types:
                    # Try first AZ with next instance type
                    current_type = fallback_types[0]
                    next_az = availability_zones[0]
                    next_subnet = get_subnet_for_az(next_az)
                else:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({
                            'RequestId': request_id,
                            'Status': 'failed',
                            'Error': 'No more configurations to try',
                            'LastError': original_error
                        })
                    }
                
                # Cancel the failed request
                ec2_client.cancel_spot_instance_requests(SpotInstanceRequestIds=[request_id])
                
                # Create new request with next configuration
                launch_spec = spot_request['LaunchSpecification']
                launch_spec.update({
                    'InstanceType': current_type,
                    'SubnetId': next_subnet
                })
                
                new_spot_response = ec2_client.request_spot_instances(
                    InstanceCount=1,
                    Type='one-time',
                    LaunchSpecification=launch_spec
                )
                
                new_request_id = new_spot_response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'RequestId': new_request_id,
                        'Status': 'pending',
                        'PreviousRequestId': request_id,
                        'CurrentInstanceType': current_type,
                        'CurrentAZ': next_az
                    })
                }
            
            elif 'InstanceId' in spot_request:
                instance_id = spot_request['InstanceId']
                
                # Check the instance status
                instance_details = ec2_client.describe_instances(InstanceIds=[instance_id])
                instance = instance_details['Reservations'][0]['Instances'][0]
                instance_state = instance['State']['Name']
                public_ip = instance.get('PublicIpAddress', 'No public IP')
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'RequestId': request_id,
                        'Status': instance_state,
                        'InstanceId': instance_id,
                        'PublicIpAddress': public_ip,
                        'InstanceType': instance['InstanceType'],
                        'AvailabilityZone': instance['Placement']['AvailabilityZone']
                    })
                }
            else:
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'RequestId': request_id,
                        'Status': 'pending'
                    })
                }

        elif event['resource'] == '/delete-machine':
            print('delete ec2 api was called...')
            
            body = parse_body(event)
            request_id = body.get('request_id')
            instance_id = body.get('instance_id')

            if request_id:
                try:
                    spot_result = ec2_client.describe_spot_instance_requests(SpotInstanceRequestIds=[request_id])
                    spot_request = spot_result['SpotInstanceRequests'][0]
                    instance_id = spot_request.get('InstanceId')
                    ec2_client.cancel_spot_instance_requests(SpotInstanceRequestIds=[request_id])
                except ClientError:
                    pass

            if instance_id:
                ec2_client.terminate_instances(InstanceIds=[instance_id])
                return {
                    'statusCode': 202,
                    'body': json.dumps({
                        'InstanceId': instance_id,
                        'RequestId': request_id,
                        'Status': 'terminating'
                    })
                }
            else:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': 'No instance ID found'
                    })
                }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
