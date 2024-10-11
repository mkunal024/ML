import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
from itertools import product

# Define a dictionary for hydrophobicity
hydrophobicity = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 1.8,
    'K': -3.9, 'L': 1.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def calculate_aac(sequences):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aac_data = []
    
    for seq in sequences:
        counts = {aa: seq.count(aa) for aa in amino_acids}
        total = len(seq)
        percentages = {aa: (count / total * 100) if total > 0 else 0 for aa, count in counts.items()}
        aac_data.append(percentages)

    return pd.DataFrame(aac_data, columns=amino_acids)

def calculate_dipeptide_composition(sequences):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    dipeptides = [''.join(p) for p in product(amino_acids, repeat=2)]
    
    dp_data = []
    
    for seq in sequences:
        counts = {dp: 0 for dp in dipeptides}
        total_dipeptides = len(seq) - 1
        
        for i in range(len(seq) - 1):
            dp = seq[i:i + 2]
            if dp in counts:
                counts[dp] += 1

        percentages = {dp: (count / total_dipeptides * 100) if total_dipeptides > 0 else 0 for dp, count in counts.items()}
        dp_data.append(percentages)

    return pd.DataFrame(dp_data, columns=dipeptides)

def calculate_tripeptide_composition(sequences):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    tripeptides = [''.join(p) for p in product(amino_acids, repeat=3)]
    
    tp_data = []
    
    for seq in sequences:
        counts = {tp: 0 for tp in tripeptides}
        total_tripeptides = len(seq) - 2
        
        for i in range(len(seq) - 2):
            tp = seq[i:i + 3]
            if tp in counts:
                counts[tp] += 1

        percentages = {tp: (count / total_tripeptides * 100) if total_tripeptides > 0 else 0 for tp, count in counts.items()}
        tp_data.append(percentages)

    return pd.DataFrame(tp_data, columns=tripeptides)

def calculate_hydrophobicity(sequences):
    hp_data = []
    
    for seq in sequences:
        total_hydrophobicity = sum(hydrophobicity.get(aa, 0) for aa in seq)
        total_length = len(seq)
        avg_hydrophobicity = total_hydrophobicity / total_length if total_length > 0 else 0
        hp_data.append({'Average_Hydrophobicity': avg_hydrophobicity})

    return pd.DataFrame(hp_data)

def calculate_molecular_weight(sequences):
    """Calculate molecular weight of each sequence."""
    weight_dict = {
        'A': 89.09, 'C': 121.15, 'D': 133.10, 'E': 147.13,
        'F': 165.19, 'G': 75.07, 'H': 155.16, 'I': 131.17,
        'K': 146.19, 'L': 131.17, 'M': 149.21, 'N': 132.12,
        'P': 115.13, 'Q': 146.15, 'R': 174.20, 'S': 105.09,
        'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
    }
    
    mw_data = []
    
    for seq in sequences:
        total_weight = sum(weight_dict.get(aa, 0) for aa in seq)
        mw_data.append({'Molecular_Weight': total_weight})

    return pd.DataFrame(mw_data)

def calculate_isoelectric_point(sequences):
    """Estimate isoelectric point (pI) based on the sequence."""
    # This is a rough estimation and may need a more sophisticated method
    # You can replace this with a more accurate prediction method if needed
    ip_data = []
    for seq in sequences:
        total_positive = sum(seq.count(aa) for aa in 'KRH')  # Basic
        total_negative = sum(seq.count(aa) for aa in 'DE')  # Acidic
        # Simple formula: pI = (total_positive - total_negative) + 7
        estimated_pI = total_positive - total_negative + 7
        ip_data.append({'Isoelectric_Point': estimated_pI})
    return pd.DataFrame(ip_data)

def main():
    # Load training and testing data
    train_data = pd.read_csv('Train_seq.csv')
    test_data = pd.read_csv('Test_seq.csv')

    # Calculate features for training data
    X_train_aac = calculate_aac(train_data['Sequence'])
    X_train_dp = calculate_dipeptide_composition(train_data['Sequence'])
    X_train_tp = calculate_tripeptide_composition(train_data['Sequence'])
    X_train_hp = calculate_hydrophobicity(train_data['Sequence'])
    X_train_mw = calculate_molecular_weight(train_data['Sequence'])
    X_train_ip = calculate_isoelectric_point(train_data['Sequence'])

    # Combine all features
    X_train = pd.concat([X_train_aac, X_train_dp, X_train_tp, X_train_hp, X_train_mw, X_train_ip], axis=1)
    y_train = train_data['Label']

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Initialize and run the TPOT AutoML classifier
    tpot = TPOTClassifier(verbosity=2, random_state=42, generations=5, population_size=20, config_dict='TPOT sparse')
    tpot.fit(X_train, y_train_encoded)

    # Print the best pipeline
    print(tpot.fitted_pipeline_)

    # Test features for test data
    X_test_aac = calculate_aac(test_data['Sequence'])
    X_test_dp = calculate_dipeptide_composition(test_data['Sequence'])
    X_test_tp = calculate_tripeptide_composition(test_data['Sequence'])
    X_test_hp = calculate_hydrophobicity(test_data['Sequence'])
    X_test_mw = calculate_molecular_weight(test_data['Sequence'])
    X_test_ip = calculate_isoelectric_point(test_data['Sequence'])

    # Combine all test features
    X_test = pd.concat([X_test_aac, X_test_dp, X_test_tp, X_test_hp, X_test_mw, X_test_ip], axis=1)
    y_test = test_data['Label']
    y_test_encoded = label_encoder.transform(y_test)

    # Predict on the test data using the best pipeline
    y_pred_encoded = tpot.predict(X_test)

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
