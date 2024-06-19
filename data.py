import pandas as pd

def load_data(filepath):
    columns = ['Compound_ID', 'Protein_ID', 'Compound_SMILES', 'Protein_Sequence', 'Affinity']
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 5:
                data.append(parts)
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    df = load_data('data/kiba.txt')
    print(df.head())
    df.to_csv('kiba_data.csv', index=False)