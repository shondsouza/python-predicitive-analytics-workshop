import pandas as pd

# s = pd.Series([10,20,30,40,50], index=["a","b","c","d","e"])
# print(s)

data = {"Name": ["Pramod", "Ravi", "Sneha"],
        "Age": [25, 30, 28],
        "Salary": [50000, 60000, 55000]}
df = pd.DataFrame(data)
print(df)

df["Name"]               
df[["Name", "Age"]]        
df.iloc[0, 1]         
df.loc[rows, cols]