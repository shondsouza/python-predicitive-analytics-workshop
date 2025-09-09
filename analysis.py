import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    'studyhours' : [1,2,3,4,5,6,7,8],
    'passed':      [0,0,0,0,1,1,1,1],
})

X = data[['study_hours']]  # Feature (independent variable)
y = data['passed']         # Target (dependent variable)

model = LogisticRegression()
model.fit(X, y)

print("Prediction (7 hours):", model.predict([[7]]))  # Expected: Pass
print("Prediction (2 hours):", model.predict([[2]]))  # Expected: Fail