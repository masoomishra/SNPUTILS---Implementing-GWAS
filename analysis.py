import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import probplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

# File paths
genotype_file_path = "/Users/masoommishra/Desktop/BME 160/Final Project SNPUTILS/mock_genotype.vcf"
phenotype_file_path = "/Users/masoommishra/Desktop/BME 160/Final Project SNPUTILS/mock_phenotype.csv"

# Load genotype data
genotype_data = pd.read_csv(genotype_file_path)

# Load phenotype data
phenotype_data = pd.read_csv(phenotype_file_path)

# Ensure both datasets match (based on sample_id)
genotype_data = genotype_data[genotype_data.sample_id.isin(phenotype_data.sample_id)]
phenotype_data = phenotype_data[phenotype_data.sample_id.isin(genotype_data.sample_id)]

# Merge the genotype and phenotype datasets
combined_data = pd.merge(genotype_data, phenotype_data, on='sample_id')

# Preprocessing: Log-transform phenotype data if it's skewed (for better model performance)
if combined_data['phenotype'].skew() > 1:  # Check for skewness
    combined_data['phenotype'] = np.log(combined_data['phenotype'])

# Process data for PCA
X = combined_data.drop(columns=['sample_id', 'phenotype'])
y = combined_data['phenotype']

# Perform PCA for dimensionality reduction (increase components)
#pca = PCA(n_components=30)  # Increase to 30 components
pca = PCA(n_components=30, random_state=42)  # <-- Added random_state=42
X_pca = pca.fit_transform(X)

np.save("X_pca.npy", X_pca)  # <-- Save PCA-transformed data

# Add constant to the PCA data for the regression
X_pca = sm.add_constant(X_pca)

# Perform OLS regression
model = sm.OLS(y, X_pca).fit()

# Display the regression summary
print(model.summary())

# Extract p-values and correct for multiple testing
p_values = model.pvalues[1:]  # Exclude constant term

# Apply multiple testing correction (FDR)
_, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

# Create Manhattan plot
plt.figure(figsize=(12, 6))
plt.scatter(range(len(corrected_p_values)), -np.log10(corrected_p_values), color='blue')
plt.xlabel('PCA Component Index')
plt.ylabel('-log10(corrected p-value)')
plt.title('Manhattan Plot')
plt.tight_layout()
plt.show()

# Create QQ plot
plt.figure(figsize=(12, 6))
probplot(-np.log10(corrected_p_values), dist="norm", plot=plt)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Observed Quantiles')
plt.title('QQ Plot')
plt.tight_layout()
plt.show()

# Correlation Matrix of PCA components
corr_matrix = pd.DataFrame(X_pca).corr()

# Plot correlation matrix as heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix (PCA Components vs Phenotype)')
plt.tight_layout()
plt.show()

print("Plots displayed successfully!")

