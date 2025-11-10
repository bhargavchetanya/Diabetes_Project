import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("diabetes.csv")
features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age']
X = data[features].to_numpy()
y = data['Outcome'].to_numpy()

mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1e-8
Xnorm = (X - mean) / std

w = np.zeros(X.shape[1])
b = 0
reg = 0.1

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def computeCost(X, y, w, b, reg):
    m, n = X.shape
    preds = sigmoid(np.dot(X, w) + b)
    totalLoss = 0
    for i in range(m):
        pred = np.clip(preds[i], 1e-9, 1-1e-9)
        totalLoss += -y[i]*np.log(pred) - (1-y[i])*np.log(1-pred)
    avgLoss = totalLoss / m
    regLoss = 0
    for j in range(n):
        regLoss += w[j]**2
    regLoss = (reg/(2*m)) * regLoss
    return avgLoss + regLoss

def computeGradient(X, y, w, b, reg):
    m, n = X.shape
    preds = sigmoid(np.dot(X, w) + b)
    dw = np.zeros(n)
    db = 0
    for i in range(m):
        error = preds[i] - y[i]
        for j in range(n):
            dw[j] += error * X[i, j]
        db += error
    for j in range(n):
        dw[j] = dw[j]/m + (reg/m)*w[j]
    db /= m
    return dw, db

def gradientDescent(X, y, w, b, alpha, iterations, reg):
    costHistory = []
    for it in range(iterations):
        dw, db = computeGradient(X, y, w, b, reg)
        w -= alpha * dw
        b -= alpha * db
        cost = computeCost(X, y, w, b, reg)
        costHistory.append(cost)
        if it % (iterations // 10) == 0 or it == iterations-1:
            print(f"Iteration {it:5}: Cost {cost:.4f}")
    return w, b, costHistory

alpha = 0.01
iterations = 2000
w, b, costHistory = gradientDescent(Xnorm, y, w, b, alpha, iterations, reg)

plt.figure(figsize=(8,5))
plt.plot(costHistory, color='purple', linewidth=2)
plt.title("Cost vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

poly = PolynomialFeatures(degree=2, include_bias=False)
Xpoly = poly.fit_transform(X[:, [0,4]])
mean_poly = Xpoly.mean(axis=0)
std_poly = Xpoly.std(axis=0)
std_poly[std_poly==0] = 1e-8
Xpoly_norm = (Xpoly - mean_poly)/std_poly

w_poly = np.zeros(Xpoly_norm.shape[1])
b_poly = 0
w_poly, b_poly, _ = gradientDescent(Xpoly_norm, y, w_poly, b_poly, alpha, iterations, reg)

plt.figure(figsize=(8,6))
x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
x2_min, x2_max = X[:,4].min()-1, X[:,4].max()+1
xx1, xx2 = np.meshgrid(np.linspace(x1_min,x1_max,200),
                       np.linspace(x2_min,x2_max,200))
grid = np.c_[xx1.ravel(), xx2.ravel()]
grid_poly = poly.transform(grid)
grid_poly_norm = (grid_poly - mean_poly)/std_poly
Z = sigmoid(np.dot(grid_poly_norm, w_poly) + b_poly)
Z = Z.reshape(xx1.shape)
plt.contour(xx1, xx2, Z, levels=[0.5], colors='green', linewidths=2)
plt.scatter(X[y==0,0], X[y==0,4], c='blue', label='No Diabetes', alpha=0.6)
plt.scatter(X[y==1,0], X[y==1,4], c='red', label='Diabetes', alpha=0.6)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('Polynomial Decision Boundary (Glucose vs BMI)')
plt.legend()
plt.grid(True)
plt.show()

#cd ~/Desktop/Diabetes_Project && pip install -r requirements.txt && python3 diabetes_logreg_from_scratch.py
