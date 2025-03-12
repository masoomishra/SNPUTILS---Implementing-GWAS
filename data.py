import pandas as pd
import numpy as np

np.random.seed(42)

# Parameters for dataset
num_samples = 500  # Number of individuals
num_snps = 10000   # Number of SNPs

# Generate random phenotype data (mean=100, std=15)
phenotype_data = pd.DataFrame({
    'sample_id': [f'sample_{i}' for i in range(1, num_samples + 1)],
    'phenotype': np.random.normal(100, 15, num_samples)  # Mean = 100, SD = 15
})

# Introduce some genetic influence on phenotype (SNP 1)
phenotype_data['phenotype'] += np.random.normal(0, 5, num_samples) * np.random.choice([0, 1, 2], num_samples)

# Generate genotype data (random SNP values: 0, 1, or 2)
genotype_data = pd.DataFrame({
    'ID': [f'rs{i}' for i in range(1, num_snps + 1)],  # SNP IDs
    **{f'sample_{i}': np.random.choice([0, 1, 2], num_snps) for i in range(1, num_samples + 1)}
})

# Transpose genotype data so that each sample is a row
genotype_data = genotype_data.set_index('ID').T.reset_index()
genotype_data.rename(columns={'index': 'sample_id'}, inplace=True)

# Save data to CSV files (updated path)
genotype_path = '/Users/masoommishra/Desktop/BME 160/Final Project SNPUTILS/mock_genotype.vcf'
phenotype_path = '/Users/masoommishra/Desktop/BME 160/Final Project SNPUTILS/mock_phenotype.csv'

genotype_data.to_csv(genotype_path, index=False)
phenotype_data.to_csv(phenotype_path, index=False)

print(f"Genotype data saved to: {genotype_path}")
print(f"Phenotype data saved to: {phenotype_path}")

