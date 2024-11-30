def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"model: {model}")
    print(f"MAE  : {round(mae, 2)}")
 



model = LinearRegression()
evaluate_model(model, x_train, y_train, x_test, y_test)




linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=44)

evaluate_model(linear_model, x_train, y_train, x_test, y_test)
evaluate_model(tree_model, x_train, y_train, x_test, y_test)




