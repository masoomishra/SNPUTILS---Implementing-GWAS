import snputils as su
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import probplot
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Genotype and Phenotype Data
# Update these paths with your actual file paths
prefix = "/Users/masoommishra/Desktop/BME 160/Final Project SNPUTILS/GWAS/project1/data/raw/all_hg38"  # This is the common prefix for .pgen, .pvar, .psam files
phenotype_file_path = "/Users/masoommishra/Desktop/BME 160/Final Project SNPUTILS/GWAS/project1/data/raw/simgwas_quant1.pheno"

# Load genotype data
genotype_data = su.read_pgen(prefix)

# Load phenotype data
phenotype_data = pd.read_csv(phenotype_file_path, sep='\t', header=None, names=['sample_id', 'phenotype'])

# Step 2: Combine Genotype and Phenotype Data
# Assuming phenotype_data has a column 'sample_id' that matches the sample IDs in genotype_data
genotype_data = genotype_data[genotype_data.sample_id.isin(phenotype_data['sample_id'])]
phenotype_data = phenotype_data[phenotype_data['sample_id'].isin(genotype_data.sample_id)]

# Merge the data
combined_data = pd.merge(genotype_data, phenotype_data, on='sample_id')

# Step 3: Perform Regression Analysis
# Example: Linear regression
X = combined_data.drop(columns=['sample_id', 'phenotype'])  # Genotype data
y = combined_data['phenotype']  # Phenotype data

# Add a constant to the predictors
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())

# Step 4: Visualization

# Extract p-values from the model
p_values = model.pvalues[1:]  # Exclude the constant term

# Create a Manhattan plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(p_values)), -np.log10(p_values), color='blue')
ax.set_xlabel('SNP Index')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Manhattan Plot')
plt.show()

# Create a QQ plot
fig, ax = plt.subplots(figsize=(10, 6))
probplot(-np.log10(p_values), dist="norm", plot=ax)
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Observed Quantiles')
ax.set_title('QQ Plot')
plt.show()

# Calculate the correlation matrix
corr_matrix = combined_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
