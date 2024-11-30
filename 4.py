
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


df=pd.read_csv("E:\\soheil\\ارشد\\term3\\machine\\HaleTamrin\\part 4\\cardio.csv")
df.head()

df['age'] = (df['age'] / 365).astype(int)
#df.columns.values[1]='age in year'
df.head()

color=['red' if c==1 else 'blue' for c in df['cardio']]
plt.scatter(df['weight'],df['height'],color=color)
plt.xlabel('weight')
plt.ylabel('height')

x = df.drop(columns=['ap_hi'])
y = df['ap_hi']

print("input dimensions :", x.shape)
print("labels dimensions:", y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

print(" data  (x_train):", x_train.shape)
print(" data  (x_test):", x_test.shape)
print(" labels  (y_train):", y_train.shape)
print(" labels(y_test):", y_test.shape)




def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"model: {model}")
    print(f"MAE  : {round(mae, 2)}")
 
 #>>   
model = LinearRegression()
evaluate_model(model, x_train, y_train, x_test, y_test)




linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=44)

evaluate_model(linear_model, x_train, y_train, x_test, y_test)
evaluate_model(tree_model, x_train, y_train, x_test, y_test)



