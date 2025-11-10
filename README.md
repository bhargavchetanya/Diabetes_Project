Diabetes Predictor (Logistic Regression from Scratch)

This project is a custom-built **Logistic Regression** model in Python to predict diabetes. It's built "from scratch" using NumPy and includes **L2 Regularization** and **Polynomial Features** to create an effective, non-linear classifier.

---

The Math

* **Sigmoid Function:** Converts model output to a probability (0 to 1).
    $$
    g(z) = \frac{1}{1 + e^{-z}}
    $$

* **Cost Function (Log Loss + L2):** Measures the model's error, which we try to minimize.
    $$
    J(\mathbf{w},b) = \frac{1}{m} \sum_{i=1}^{m} \left[ -y^{(i)} \log(g(z^{(i)})) - (1-y^{(i)}) \log(1-g(z^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
    $$

* **Gradient:** The "direction" of the steepest cost. Used by gradient descent to update $\mathbf{w}$ and $b$.
    $$
    \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (g(z^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j
    $$

---

Model Output

* **Cost vs. Iterations:** Shows the cost decreasing as the model learns.
    

* **Decision Boundary:** The final curved line the model learned to separate "Diabetes" (red) from "No Diabetes" (blue) using Glucose and BMI.
    

---

## 🚀 How to Run

1.  Ensure you have `diabetes.csv` in the same folder.
2.  Install requirements: `pip install -r requirements.txt`
3.  Run the script: `python3 diabetes_logreg_from_scratch.py`
