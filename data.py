import pandas as pd

def load_data(filepath):
    columns = ['Compound_ID', 'Protein_ID', 'Compound_SMILES', 'Protein_Sequence', 'Affinity']
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 5:
                data.append(parts)
    df = pd.DataFrame(data, columns=columns)
    
    # 只保留前 85% 的数据
    cutoff = int(len(df) * 0.85)
    df = df.iloc[:cutoff]
    
    return df

if __name__ == "__main__":
    df = load_data('data/kiba.txt')
    print(df.head())
    df.to_csv('kiba_data.csv', index=False)