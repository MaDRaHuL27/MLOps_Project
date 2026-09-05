# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# Load the dataset that was registered into tourism_project/data/
tourism_dataset = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

# --- Data Cleaning ---
# Drop identifier / index columns that carry no predictive signal
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in tourism_dataset.columns]
tourism_dataset = tourism_dataset.drop(columns=drop_cols)

# Fix a known data-entry typo in Gender ("Fe Male" -> "Female") so the
# category isn't split into two separate levels during one-hot encoding
tourism_dataset["Gender"] = tourism_dataset["Gender"].replace({"Fe Male": "Female"})

print("Columns after cleaning:", list(tourism_dataset.columns))

# Define the target variable for the classification task
target = "ProdTaken"

# List of numerical features in the dataset
numeric_features = [
    "Age",                       # Customer's age
    "CityTier",                  # City category (1 > 2 > 3)
    "DurationOfPitch",           # Duration of the sales pitch
    "NumberOfPersonVisiting",    # Total people accompanying the customer
    "NumberOfFollowups",         # Number of follow-ups by the salesperson
    "PreferredPropertyStar",     # Preferred hotel rating
    "NumberOfTrips",             # Average annual number of trips
    "Passport",                  # Holds a valid passport (0/1)
    "PitchSatisfactionScore",    # Satisfaction score with the sales pitch
    "OwnCar",                    # Owns a car (0/1)
    "NumberOfChildrenVisiting",  # Number of children (<5) accompanying
    "MonthlyIncome",             # Gross monthly income
]

# List of categorical features in the dataset
categorical_features = [
    "TypeofContact",   # Company Invited or Self Enquiry
    "Occupation",       # Salaried, Freelancer, etc.
    "Gender",            # Male / Female
    "ProductPitched",   # Product pitched to the customer
    "MaritalStatus",     # Single, Married, Divorced, Unmarried
    "Designation",       # Customer's designation at their organisation
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]

# Split dataset into train and test
# stratify=y keeps the (imbalanced) ProdTaken ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Xtrain shape: {Xtrain.shape}, Xtest shape: {Xtest.shape}")
