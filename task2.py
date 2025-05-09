import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Test if script is running
print("Script is running!")

# Step 1: Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Check if the data is loaded correctly
print("\nDataset loaded successfully!")
print(df.head())  # Print the first few rows of the dataframe

# Step 2: Drop non-numeric columns
df_numeric = df.drop(columns=['CustomerID', 'Gender'])
print("\nNon-numeric columns dropped.")
print(df_numeric.head())  # Print the first few rows after dropping non-numeric columns

# Step 3: Standardize the numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)
print("\nData standardized.")

# Step 4: Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("\nK-means clustering applied.")

# Step 5: Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2',
    s=100,
    edgecolor='black'
)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)

# Save the scatter plot as an image
plt.savefig('customer_segmentation.png')
plt.show()

# Step 6: Elbow Method to determine optimal k
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid(True)

# Save the elbow plot as an image
plt.savefig('elbow_method.png')
plt.show()

# Step 7: Print sample of clustered data
print("\nSample of clustered data:")
print(df.head())

# Step 8: Print customer count in each cluster
print("\nCustomer count in each cluster:")
print(df['Cluster'].value_counts())

# Pause before closing
input("\nPress Enter to exit...")
