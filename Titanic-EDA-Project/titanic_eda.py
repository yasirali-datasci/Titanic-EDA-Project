# 📌 Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 Load Titanic dataset (train.csv)
df = pd.read_csv("train.csv")

# 📌 Show total missing values in each column
print(df.isnull().sum())

# 📌 Specifically check missing values in 'Age' column
print("\nMissing values in Age column: ", df["Age"].isnull().sum())

# 📌 Calculate mean age (ignoring NaN)
mean_age = df["Age"].mean()
print("Mean of Age: ", mean_age)

# 📌 Fill missing 'Age' values with mean age
df["Age"] = df["Age"].fillna(mean_age)

# 📌 Confirm missing 'Age' values have been filled
print("\nMissing Age values after filling: ", df["Age"].isnull().sum())

# 📌 Boxplot: Age distribution vs Survived (0 = No, 1 = Yes)
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survived - Boxplot")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

# 📌 Boxplot: Age distribution vs Survived, separated by Gender
sns.boxplot(x="Survived", y="Age", hue="Sex", data=df)
plt.title("Age vs Survived by Gender")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

# 📌 Correlation Matrix: check relationships between Survived, Age, Fare, Pclass
corr = df[["Survived", "Age", "Fare", "Pclass"]].corr()
print("\nCorrelation Matrix:\n", corr)

# 📌 Heatmap to visualize correlation matrix
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
