import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set up the page
st.set_page_config(page_title="Iris Decision Tree Classifier", layout="wide")
st.title("🌸 Iris Species Classifier Using Decision Tree")

# Load Iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

df, iris = load_data()

# Show the dataset
st.subheader("📊 Iris Dataset Preview")
st.dataframe(df.head())

# Sidebar - model parameters
st.sidebar.header("🔧 Model Parameters")
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

# Split data
X = df[iris.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
st.subheader("✅ Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.2%}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# Feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = np.array(iris.feature_names)

fig_imp, ax_imp = plt.subplots()
sns.barplot(x=importances[indices], y=features[indices], palette="viridis", ax=ax_imp)
ax_imp.set_title("Feature Importances")
st.pyplot(fig_imp)

# Decision Tree Plot
st.subheader("🌳 Visualized Decision Tree")
fig_tree, ax_tree = plt.subplots(figsize=(16, 10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, ax=ax_tree)
st.pyplot(fig_tree)

# Footer
# st.markdown("---")
# st.caption("Created with ❤️ using Streamlit and Scikit-learn")
