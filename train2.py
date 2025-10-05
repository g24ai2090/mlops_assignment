from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_model, evaluate_model

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)
model = KernelRidge(alpha=1.0)  # Default params; tune if needed
trained_model = train_model(model, X_train, y_train)
mse = evaluate_model(trained_model, X_test, y_test)
print(f"Average MSE on test set: {mse}")