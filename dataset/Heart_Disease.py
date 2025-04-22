# Import the fetch_ucirepo function from the ucimlrepo library
from ucimlrepo import fetch_ucirepo 

# Fetch the dataset with ID 45 (Heart Disease dataset)
heart_disease = fetch_ucirepo(id=45) 

# Extract the features (independent variables) and targets (dependent variable) as pandas DataFrames
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Print metadata about the dataset (e.g., description, source, etc.)
print(heart_disease.metadata) 

# Print information about the variables (e.g., feature names, types, etc.)
print(heart_disease.variables)