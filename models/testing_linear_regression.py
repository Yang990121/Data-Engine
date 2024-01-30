from sklearn.linear_model import LinearRegression

class SimpleLinearRegressionModel:
    def __init__(self, df):
        self.df = df

    def predict(self, selected_column):
        X = self.df.drop(columns=[selected_column])
        y = self.df[selected_column]

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        return predictions
