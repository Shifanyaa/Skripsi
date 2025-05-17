import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from IPython.display import display

# Load train set (after SMOTE)
train_df = pd.read_csv('train.csv')
features = train_df.columns.drop('Label')
X_train = train_df[features]
y_train = train_df['Label']

# 1. Korelasi fitur
df_corr = X_train.corr()
# cari pasangan dengan abs(corr) > 0.9, selain diagonal
dup_pairs = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        corr_val = df_corr.iloc[i, j]
        if abs(corr_val) > 0.9:
            dup_pairs.append({
                'Feature1': features[i],
                'Feature2': features[j],
                'Correlation': corr_val
            })
high_corr_df = pd.DataFrame(dup_pairs)

# Display results
print("High Correlation Pairs (>0.9):")
display(high_corr_df)

# 2. Visualisasi distribusi per fitur
for feature in features:
    # Histogram per kelas
    plt.figure()
    plt.hist(X_train.loc[y_train==0, feature], bins=30, density=True, alpha=0.5, label='Class 0')
    plt.hist(X_train.loc[y_train==1, feature], bins=30, density=True, alpha=0.5, label='Class 1')
    plt.title(f'Histogram of {feature} by Class')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Boxplot per kelas
    plt.figure()
    data_to_plot = [X_train.loc[y_train==0, feature], X_train.loc[y_train==1, feature]]
    plt.boxplot(data_to_plot, labels=['Class 0', 'Class 1'])
    plt.title(f'Boxplot of {feature} by Class')
    plt.ylabel(feature)
    plt.show()

# 3. Validasi SMOTE dengan t-SNE visualisasi
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_train)

plt.figure()
plt.scatter(X_embedded[y_train==0, 0], X_embedded[y_train==0, 1], alpha=0.5, label='Class 0')
plt.scatter(X_embedded[y_train==1, 0], X_embedded[y_train==1, 1], alpha=0.5, label='Class 1')
plt.legend()
plt.title('t-SNE of Train Data After SMOTE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# 4. Cek variansi fitur
variances = X_train.var()
low_var = variances[variances < 0.01].reset_index()
low_var.columns = ['Feature', 'Variance']

print("\nLow Variance Features (<0.01):")
display(low_var)
