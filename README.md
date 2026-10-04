# EX NO. 3(b)

# DATA PREPROCESSING

## AIM

To perform data preprocessing on a given dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset into training and testing sets, and applying feature scaling.

---

# OBJECTIVES

The main objectives of this experiment are:

1. To understand the concept of data preprocessing.
2. To load a dataset using Python and Pandas.
3. To inspect the structure and characteristics of the dataset.
4. To identify independent and dependent variables.
5. To identify and handle missing values.
6. To encode categorical variables.
7. To apply Label Encoding.
8. To apply One-Hot Encoding.
9. To divide the dataset into training and testing sets.
10. To apply feature scaling using StandardScaler.
11. To prepare the dataset for machine learning algorithms.

---

# INTRODUCTION

Data preprocessing is an important step in the data science and machine learning process.

Raw datasets may contain missing values, categorical variables, different numerical ranges, and other inconsistencies. Machine learning algorithms generally require data to be converted into a suitable numerical format before training.

Data preprocessing transforms raw data into a clean and structured form that can be efficiently used by machine learning models.

The major steps involved in data preprocessing are:

**Data Collection → Data Cleaning → Data Encoding → Data Splitting → Feature Scaling → Machine Learning**

In this experiment, a customer dataset containing the attributes **Country, Age, Salary, and Purchased** is used.

The dataset contains both numerical and categorical information. Therefore, preprocessing techniques are applied to prepare the dataset for machine learning.

---

# THEORY

## 1. What is Data Preprocessing?

Data preprocessing is the process of converting raw data into a suitable format for analysis and machine learning.

Real-world datasets may contain:

* Missing values
* Categorical data
* Numerical data
* Duplicate records
* Different scales
* Incorrect or inconsistent values

If these problems are not handled properly, they may reduce the performance of a machine learning model.

Therefore, data preprocessing is performed before applying a machine learning algorithm.

---

# 2. IMPORTANCE OF DATA PREPROCESSING

Data preprocessing is important because it:

1. Improves data quality.
2. Handles missing values.
3. Converts categorical data into numerical form.
4. Reduces inconsistencies.
5. Makes data suitable for machine learning.
6. Improves model performance.
7. Prevents numerical-scale problems.
8. Makes the dataset easier to analyze.
9. Helps machine learning algorithms process data efficiently.
10. Provides standardized input features.

---

# 3. DATASET USED

The dataset used in this experiment contains customer information.

The main attributes are:

| Attribute | Description             | Type        |
| --------- | ----------------------- | ----------- |
| Country   | Country of the customer | Categorical |
| Age       | Age of the customer     | Numerical   |
| Salary    | Salary of the customer  | Numerical   |
| Purchased | Purchasing decision     | Categorical |

The independent variables are:

**Country, Age, Salary**

The dependent variable is:

**Purchased**

The dependent variable represents the target that the machine learning model attempts to predict.

---

# 4. INDEPENDENT VARIABLES

Independent variables are the input features used to predict the target variable.

In this experiment:

**X = Country, Age, Salary**

These features provide information about the customer.

For example:

* Country provides geographical information.
* Age provides demographic information.
* Salary provides financial information.

---

# 5. DEPENDENT VARIABLE

The dependent variable is the output or target variable.

In this experiment:

**Y = Purchased**

The Purchased attribute indicates whether the customer purchased the product.

It contains categorical values such as:

* Yes
* No

Since machine learning algorithms generally work with numerical values, this variable is converted into numerical form using Label Encoding.

---

# 6. PYTHON FOR DATA PREPROCESSING

Python provides several libraries that are useful for data preprocessing.

Important libraries include:

* Pandas
* NumPy
* Scikit-learn

### Pandas

Pandas is used for loading and manipulating datasets.

### NumPy

NumPy is used for numerical operations and array processing.

### Scikit-learn

Scikit-learn provides preprocessing and machine learning functions such as:

* SimpleImputer
* LabelEncoder
* OneHotEncoder
* train_test_split
* StandardScaler

---

# 7. GOOGLE COLAB

Google Colab is a cloud-based platform that allows Python programs to be executed through a web browser.

It is useful for machine learning and data science experiments because it provides:

* Python environment
* Notebook interface
* Data analysis libraries
* Machine learning libraries
* Google Drive integration

In this experiment, Google Drive is mounted to access the dataset.

---

# 8. IMPORTING REQUIRED LIBRARIES

The required libraries are imported before preprocessing.

```python
import pandas as pd
import numpy as np
```

Pandas is imported as `pd`.

NumPy is imported as `np`.

These libraries provide functions required for data handling and numerical operations.

---

# 9. LOADING THE DATASET

The dataset is loaded using the Pandas `read_csv()` function.

```python
df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')
```

The CSV file is converted into a Pandas DataFrame.

The DataFrame is stored in the variable:

```python
df
```

---

# 10. DISPLAYING THE DATASET

The first few records of the dataset can be displayed using:

```python
df.head()
```

The `head()` function displays the first five rows by default.

It helps us understand the structure and values present in the dataset.

---

# 11. INSPECTING THE DATASET

The `info()` function is used to obtain information about the dataset.

```python
df.info()
```

It displays:

* Number of entries
* Column names
* Non-null values
* Data types
* Memory usage

The shape of the dataset can be obtained using:

```python
print(df.shape)
```

For the given dataset:

**Number of records = 10**

**Number of columns = 4**

Therefore:

**Shape = (10, 4)**

---

# 12. SEPARATING INDEPENDENT AND DEPENDENT VARIABLES

The independent variables are separated from the dependent variable.

```python
x = df[['Country', 'Age', 'Salary']]
y = df[['Purchased']].values
```

Here:

`x` contains the input features.

`y` contains the target variable.

The independent variables are:

**Country, Age, Salary**

The dependent variable is:

**Purchased**

---

# 13. CONVERTING DATA INTO AN ARRAY

Scikit-learn preprocessing functions can work with NumPy arrays.

Therefore, the independent variables are converted into an array.

```python
x = df[['Country', 'Age', 'Salary']].values
```

The `.values` property converts the Pandas DataFrame into a NumPy array.

The resulting array contains the values of Country, Age, and Salary.

---

# 14. MISSING VALUES

A missing value occurs when a particular data field does not contain a valid value.

For example:

| Country | Age | Salary | Purchased |
| ------- | --: | -----: | --------- |
| France  |  44 |  72000 | No        |
| Spain   |  27 |  48000 | Yes       |
| Germany |  30 |    NaN | No        |

In this example, the Salary value is missing.

Missing values can create problems for machine learning algorithms.

Therefore, they must be handled before training a model.

---

# 15. HANDLING MISSING VALUES USING SIMPLEIMPUTER

Scikit-learn provides the `SimpleImputer` class for handling missing values.

The following code is used:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
)
```

The strategy used in this experiment is:

**Mean Strategy**

The mean value of the available numerical values is calculated and used to replace missing values.

---

# 16. APPLYING THE IMPUTER

The imputer is applied only to the numerical columns.

```python
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
```

The expression:

```python
x[:, 1:3]
```

selects the Age and Salary columns.

The missing values in these columns are replaced by their corresponding mean values.

This helps maintain the dataset without removing the entire record.

---

# 17. CATEGORICAL DATA

Categorical data contains labels or categories instead of numerical values.

In the given dataset:

**Country** is categorical.

Examples of country values include:

* France
* Spain
* Germany

Machine learning algorithms generally require numerical input.

Therefore, categorical data must be converted into numerical representation.

---

# 18. LABEL ENCODING

Label Encoding converts categorical values into numerical labels.

Scikit-learn provides the `LabelEncoder` class.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])
```

The Country column is encoded into numerical values.

For example, the categories may be represented internally by different integer labels.

The exact numerical labels depend on the categories and their ordering.

---

# 19. ADVANTAGE OF LABEL ENCODING

Label Encoding is simple and easy to implement.

It is useful when:

* Categories have an inherent order.
* A categorical variable needs to be converted into integers.
* The number of categories is small.

However, for nominal categories such as countries, directly assigning integer values may imply an artificial ordering.

Therefore, One-Hot Encoding is commonly used for nominal categorical features.

---

# 20. ONE-HOT ENCODING

One-Hot Encoding converts categorical values into separate binary columns.

For example, if the Country column contains:

* France
* Germany
* Spain

One-Hot Encoding creates separate columns for these categories.

Conceptually:

| Country | France | Germany | Spain |
| ------- | -----: | ------: | ----: |
| France  |      1 |       0 |     0 |
| Germany |      0 |       1 |     0 |
| Spain   |      0 |       0 |     1 |

Each row contains a value of 1 for its corresponding category and 0 for the other categories.

---

# 21. IMPLEMENTING ONE-HOT ENCODING

The `OneHotEncoder` class from Scikit-learn is used.

```python
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x_country = onehotencoder.fit_transform(
    df.Country.values.reshape(-1, 1)
).toarray()

print(x_country)
```

The Country column is reshaped into a two-dimensional array.

The encoder then creates binary columns for the different country categories.

---

# 22. ENCODING THE DEPENDENT VARIABLE

The dependent variable is:

**Purchased**

It contains categorical values such as:

* Yes
* No

The dependent variable is encoded using LabelEncoder.

```python
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
```

After encoding, the categorical target values are converted into numerical labels.

For example, the two categories can be represented by two numerical values.

The actual assignment is determined by the encoder.

---

# 23. TRAINING AND TESTING DATA

After preprocessing, the dataset is divided into two parts:

1. Training dataset
2. Testing dataset

The training dataset is used to train the machine learning model.

The testing dataset is used to evaluate the model on previously unseen data.

This separation helps determine whether the model can generalize beyond the training data.

---

# 24. TRAIN-TEST SPLIT

Scikit-learn provides the `train_test_split()` function.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)
```

In this experiment:

**Test size = 0.2**

This means approximately 20% of the data is allocated for testing and the remaining 80% is allocated for training.

For a dataset containing 10 records, this results in approximately:

**Training data = 8 records**

**Testing data = 2 records**

---

# 25. RANDOM STATE

The `random_state` parameter is used to obtain reproducible results.

```python
random_state=0
```

Using a fixed random state ensures that the same split can be reproduced when the program is executed again.

---

# 26. FEATURE SCALING

Feature scaling is the process of transforming numerical features so that they have comparable scales.

Consider the following features:

* Age
* Salary

Age may have values in tens, while Salary may have values in thousands.

Therefore, the numerical ranges are significantly different.

Some machine learning algorithms can be affected by these differences in scale.

Feature scaling helps solve this problem.

---

# 27. STANDARDIZATION

Standardization transforms features so that they are centered around a mean of approximately zero with a standard deviation of approximately one.

Scikit-learn provides the `StandardScaler` class.

```python
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
```

The scaler is fitted using the training data:

```python
x_train = sc_x.fit_transform(x_train)
```

The same transformation is then applied to the test data:

```python
x_test = sc_x.transform(x_test)
```

The test data must be transformed using the scaler fitted on the training data to avoid using information from the test set during preprocessing.

---

# 28. COMPLETE PROGRAM

```python
# Step 1: Import libraries and load dataset

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')

# Display dataset
print(df.head())

# Step 2: Check dataset information

df.info()
print(df.shape)

# Step 3: Separate independent and dependent variables

x = df[['Country', 'Age', 'Salary']]
y = df[['Purchased']].values

# Convert X into array
x = df[['Country', 'Age', 'Salary']].values

# Step 4: Handle missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
)

imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

# Step 5: Encode categorical data

from sklearn.preprocessing import LabelEncoder

label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

print(x)

# Step 6: One-Hot Encoding

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x_country = onehotencoder.fit_transform(
    df.Country.values.reshape(-1, 1)
).toarray()

print(x_country)

# Encode dependent variable

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print(y)

# Step 7: Split dataset into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

print("Training Data:")
print(x_train)

print("Testing Data:")
print(x_test)

print("Training Target:")
print(y_train)

print("Testing Target:")
print(y_test)

# Step 8: Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print("Scaled Training Data:")
print(x_train)

print("Scaled Testing Data:")
print(x_test)
```

---

# 29. ALGORITHM

### Step 1

Start the program.

### Step 2

Import Pandas and NumPy.

### Step 3

Mount Google Drive and load the CSV dataset.

### Step 4

Display the first few records using `head()`.

### Step 5

Inspect the dataset using `info()` and `shape`.

### Step 6

Separate independent variables and the dependent variable.

### Step 7

Convert the independent variables into a NumPy array.

### Step 8

Identify missing numerical values.

### Step 9

Replace missing numerical values using the mean strategy of `SimpleImputer`.

### Step 10

Apply Label Encoding to the Country column.

### Step 11

Apply One-Hot Encoding to the Country column.

### Step 12

Apply Label Encoding to the Purchased column.

### Step 13

Split the dataset into training and testing sets.

### Step 14

Apply StandardScaler to the training data.

### Step 15

Transform the testing data using the fitted scaler.

### Step 16

Display the preprocessed training and testing data.

### Step 17

Stop the program.

---

# 30. FLOW OF DATA PREPROCESSING

**Raw Dataset**

↓

**Load Dataset Using Pandas**

↓

**Inspect Dataset**

↓

**Separate X and Y**

↓

**Handle Missing Values**

↓

**Encode Categorical Variables**

↓

**Split Training and Testing Data**

↓

**Feature Scaling**

↓

**Preprocessed Dataset**

↓

**Machine Learning Model**

---

# 31. OBSERVATION

The following observations are obtained after preprocessing:

| Step | Operation          | Purpose                          |
| ---- | ------------------ | -------------------------------- |
| 1    | Load dataset       | Read the input data              |
| 2    | `head()`           | View initial records             |
| 3    | `info()`           | Inspect data structure           |
| 4    | `shape`            | Identify rows and columns        |
| 5    | X and Y separation | Identify features and target     |
| 6    | SimpleImputer      | Handle missing values            |
| 7    | LabelEncoder       | Convert categories into labels   |
| 8    | OneHotEncoder      | Create binary category features  |
| 9    | train_test_split   | Create training and testing sets |
| 10   | StandardScaler     | Scale numerical features         |

---

# 32. EXPECTED OUTPUT

After executing the program, the following outputs are obtained:

### Dataset Preview

The first five records of the customer dataset are displayed.

### Dataset Information

The dataset information including the number of entries, column names, non-null values, and data types is displayed.

### Dataset Shape

The shape of the dataset is displayed as:

**(10, 4)**

### After Missing Value Handling

Missing numerical values in the Age and Salary columns are replaced using their mean values.

### After Encoding

The Country and Purchased columns are converted into numerical representations.

### Training and Testing Data

The dataset is divided into approximately:

**80% Training Data**

**20% Testing Data**

### After Feature Scaling

The training and testing features are transformed using StandardScaler.

---

# OUTPUT

The dataset was successfully loaded and preprocessed using Python, Pandas, NumPy, and Scikit-learn.

The preprocessing operations successfully performed were:

* Dataset loading
* Dataset inspection
* Independent and dependent variable separation
* Conversion into arrays
* Missing value handling using `SimpleImputer`
* Label Encoding
* One-Hot Encoding
* Target variable encoding
* Training and testing data splitting
* Feature scaling using `StandardScaler`

The preprocessed training and testing datasets were successfully displayed.

---

# RESULT

Thus, the given dataset was successfully preprocessed using Python and Scikit-learn.

Missing numerical values were handled using the mean strategy, categorical variables were encoded, the dataset was divided into training and testing sets, and feature scaling was performed using StandardScaler.

The resulting dataset is prepared for use in subsequent machine learning algorithms.

---

# CONCLUSION

Data preprocessing is an essential stage in machine learning because raw data often cannot be directly supplied to a machine learning algorithm.

In this experiment, the customer dataset containing **Country, Age, Salary, and Purchased** attributes was successfully preprocessed.

Missing numerical values were handled using **SimpleImputer with the mean strategy**. Categorical data was converted into numerical form using **LabelEncoder and One-Hot Encoding**. The dependent variable was also encoded using LabelEncoder.

The dataset was then divided into training and testing sets using `train_test_split()`. Finally, feature scaling was performed using **StandardScaler**.

Therefore, the experiment successfully demonstrated the major preprocessing techniques required to convert raw data into a suitable format for machine learning.

