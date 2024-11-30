from sklearn.preprocessing import LabelEncoder

sample_label_encoder = LabelEncoder()
sample_labels_encoded = sample_label_encoder.fit_transform(sample_labels)
print(set(sample_labels_encoded))


#from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(path, valid_lead=['MLII'], fs_out=360):
    all_data = []
    all_labels = []

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    for record_name in all_record_name:
        try:
            tmp_ann_res = wfdb.rdann(os.path.join(path, record_name), 'atr')
            tmp_data_res = wfdb.rdsamp(os.path.join(path, record_name))
        except:
            print(f'Error reading {record_name}')
            continue

        fs = tmp_data_res[1]['fs']
        lead_in_data = tmp_data_res[1]['sig_name']

        if valid_lead[0] in lead_in_data:
            channel = lead_in_data.index(valid_lead[0])
            tmp_data = tmp_data_res[0][:, channel]

            idx_list = list(tmp_ann_res.sample)
            label_list = tmp_ann_res.symbol

            for i in range(len(label_list)):
                s = label_list[i]
                if s in label_group_map.keys():
                    idx_start = idx_list[i] - int(fs / 2)
                    idx_end = idx_list[i] + int(fs / 2)
                    if 0 <= idx_start < len(tmp_data) and idx_end <= len(tmp_data):
                        segment = tmp_data[idx_start:idx_end]
                        segment_resampled = resample_interpolation(segment, fs, fs_out)
                        all_data.append(segment_resampled)
                        all_labels.append(label_group_map[s])

            print(f'Processed {record_name}, total segments: {len(all_data)}')

    label_encoder = LabelEncoder()
    all_labels_encoded = label_encoder.fit_transform(all_labels)

    return np.array(all_data), all_labels_encoded

path = r'------------'
all_data, all_labels = load_and_preprocess_data(path)
print()
print(f'Preprocessed data shape: {all_data.shape}')
print(f'Preprocessed labels shape: {all_labels.shape}')





import numpy as np

unique_labels, counts = np.unique(all_labels, return_counts=True)

for label, count in zip(unique_labels, counts):
    print(f'Class {label}: {count} instances')






import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(records):
    features = []
    for record in records:
        mean_val = np.mean(record)
        std_val = np.std(record)
        max_val = np.max(record)
        min_val = np.min(record)
        
        skewness = skew(record)
        kurt = kurtosis(record)
        
        sorted_neg_values = np.sort(record[record < 0])
        second_min_val = sorted_neg_values[1] if len(sorted_neg_values) > 1 else (sorted_neg_values[0] if len(sorted_neg_values) == 1 else np.nan)
        
        fft_vals = np.fft.fft(record)
        power_spectrum = np.mean(np.abs(fft_vals) ** 2)

        features.append([mean_val, std_val, max_val, min_val, skewness, kurt, second_min_val, power_spectrum])
    
    return np.array(features)

features = extract_features(all_data)

print(f'Extracted features shape: {features.shape}')
print(features[:5])









from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, all_labels, test_size=0.2, random_state=42)

print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
print(f'Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}')









from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def train_ensemble_voting(X_train, y_train):
    imputer = SimpleImputer(strategy='mean')
    
    svm_model = Pipeline([
        ('imputer', imputer),
        ('classifier', SVC(probability=True, random_state=44, class_weight='balanced'))
    ])
    
    rf_model = Pipeline([
        ('imputer', imputer),
        ('classifier', RandomForestClassifier(random_state=44, class_weight='balanced'))
    ])
    
    ensemble_model = VotingClassifier(estimators=[
        ('svm', svm_model),
        ('rf', rf_model)], voting='soft')

    ensemble_model.fit(X_train, y_train)
    return ensemble_model

ensemble_model = train_ensemble_voting(X_train, y_train)

print('Ensemble model trained successfully')






from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = ensemble_model.predict(X_test)

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

evaluate_model(y_test, y_pred)










