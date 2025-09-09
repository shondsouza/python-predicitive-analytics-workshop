import seaborn as sns           # Seaborn comes with some built-in sample datasets (like Titanic, Iris, Tips, Penguins, etc.)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = sns.load_dataset("titanic").dropna(subset=['age','fare','class','sex','survived'])

df['sex'] = df['sex'].map({'male':0, 'female':1})
X = df[['age','fare','sex','pclass']]
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)      # y= f(x)
print("Accuracy:", accuracy_score(y_test, y_pred))