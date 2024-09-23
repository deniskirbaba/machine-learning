Главное отличие параметров от гиперпараметров - зависимость от данных.  
Параметры модели зависят от данных, в то время как гиперпараметры - нет.

При разбиении датасета на две составляющие: train и test.  
При подборе гиперпараметров на train выборке и оценке их эффективности на test выборке будет происходить `утечка данных`. Она происходит из-за того что мы, как эксперт будем являться "методом оптимизации" и просто-напросто подбираем лучшие гиперпараметры под тестовую выборку. Из-за этого будет наблюдаться переобучение на гиперпараметрах.
Чтобы избежать этого самым простым решением будет выделение еще одной части из датасета.  
То есть теперть будем разбивать: train, validation, test.

Регуляризация в целом - это такие "искусственные" ограничения на нашу модель, которые стабилизируют решение.

Градиентный спуск - итеративный метод поиска минимума функции потерь. Имеет меньшую вычислительную сложность, нежели аналитическое решение. Стохастический градиентный спуск использует не всю выборку при расчете градиента, а только некоторое подмножество. Таким образом, для поиска минимума обычному градиентному спуску потребуется меньше шагов, чем стохастическому. Однако из-за того, что скорость вычисления шага значительно меньше у SGD, то общее время поиска минимума меньше.

# Matrix differentiation

### Basic Notation
- **Vectors** are considered column vectors unless otherwise noted.
- $ \mathbf{x} \in \mathbb{R}^n $ is a vector.
- $ \mathbf{A} \in \mathbb{R}^{m \times n} $ is a matrix.
- $ f(\mathbf{x}) $ is a scalar function of a vector $ \mathbf{x} $.

### 1. **Gradient of a Scalar Function** $ f(\mathbf{x}) $
The gradient of a scalar function $ f(\mathbf{x}) $, where $ \mathbf{x} \in \mathbb{R}^n $, is a **vector** of partial derivatives:
$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n
$$

### 2. **Jacobian of a Vector Function** $ \mathbf{f}(\mathbf{x}) $
For a vector-valued function $ \mathbf{f}(\mathbf{x}) \in \mathbb{R}^m $, the **Jacobian** matrix contains all first-order partial derivatives of $ \mathbf{f} $ with respect to $ \mathbf{x} $:
$$
\mathbf{J}_{\mathbf{f}}(\mathbf{x}) = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}
$$

### 3. **Hessian of a Scalar Function** $ f(\mathbf{x}) $
The Hessian is the matrix of second-order partial derivatives of a scalar function $ f(\mathbf{x}) $, where $ \mathbf{x} \in \mathbb{R}^n $:
$$
\mathbf{H}_f(\mathbf{x}) = \frac{\partial^2 f}{\partial \mathbf{x}^2} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix} \in \mathbb{R}^{n \times n}
$$

### 4. **Matrix Differentiation Rules**

#### a. **Constant Matrix**
If $ \mathbf{A} \in \mathbb{R}^{m \times n} $ is a constant matrix, then the derivative of a constant matrix is zero:
$$
\frac{\partial \mathbf{A}}{\partial \mathbf{x}} = 0
$$

#### b. **Linear Function** $ f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x} $
For a linear function $ f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x} $, where $ \mathbf{a} \in \mathbb{R}^n $, the derivative is:
$$
\frac{\partial}{\partial \mathbf{x}} \left( \mathbf{a}^\top \mathbf{x} \right) = \mathbf{a}
$$

#### c. **Quadratic Form** $ f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x} $
For a quadratic form $ f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x} $, where $ \mathbf{A} \in \mathbb{R}^{n \times n} $ is symmetric, the derivative is:
$$
\frac{\partial}{\partial \mathbf{x}} \left( \mathbf{x}^\top \mathbf{A} \mathbf{x} \right) = 2 \mathbf{A} \mathbf{x}
$$

#### d. **Derivative of $ \mathbf{a}^\top \mathbf{X} \mathbf{b} $** (Bilinear Form)
For $ f(\mathbf{X}) = \mathbf{a}^\top \mathbf{X} \mathbf{b} $, where $ \mathbf{X} \in \mathbb{R}^{n \times m} $, $ \mathbf{a} \in \mathbb{R}^n $, and $ \mathbf{b} \in \mathbb{R}^m $, the derivative is:
$$
\frac{\partial}{\partial \mathbf{X}} \left( \mathbf{a}^\top \mathbf{X} \mathbf{b} \right) = \mathbf{a} \mathbf{b}^\top
$$

#### e. **Trace of a Matrix Product** $ f(\mathbf{X}) = \text{Tr}(\mathbf{A}^\top \mathbf{X}) $
For the trace of a matrix product, where $ \mathbf{A} \in \mathbb{R}^{n \times m} $ and $ \mathbf{X} \in \mathbb{R}^{n \times m} $, the derivative is:
$$
\frac{\partial}{\partial \mathbf{X}} \text{Tr}(\mathbf{A}^\top \mathbf{X}) = \mathbf{A}
$$

#### f. **Derivative of the Determinant** $ f(\mathbf{X}) = \det(\mathbf{X}) $
For the determinant of a square matrix $ \mathbf{X} \in \mathbb{R}^{n \times n} $, the derivative is:
$$
\frac{\partial}{\partial \mathbf{X}} \det(\mathbf{X}) = \det(\mathbf{X}) \mathbf{X}^{-\top}
$$

#### g. **Matrix Inverse** $ f(\mathbf{X}) = \mathbf{X}^{-1} $
For the matrix inverse $ \mathbf{X}^{-1} $, the derivative is:
$$
\frac{\partial}{\partial \mathbf{X}} \mathbf{X}^{-1} = -\mathbf{X}^{-1} \otimes \mathbf{X}^{-\top}
$$
where $ \otimes $ denotes the Kronecker product.

---

### Summary of Basic Formulas

| Function $ f(\mathbf{x}) $ or $ f(\mathbf{X}) $ | Derivative |
| ---------------------------------------- | ------------------------------------------------- |
| $ \mathbf{a}^\top \mathbf{x} $         | $ \mathbf{a} $ |
| $ \mathbf{x}^\top \mathbf{A} \mathbf{x} $  | $ 2 \mathbf{A} \mathbf{x} $ (if $ \mathbf{A} $ is symmetric) |
| $ \mathbf{a}^\top \mathbf{X} \mathbf{b} $ | $ \mathbf{a} \mathbf{b}^\top $ |
| $ \text{Tr}(\mathbf{A}^\top \mathbf{X}) $ | $ \mathbf{A} $ |
| $ \det(\mathbf{X}) $ | $ \det(\mathbf{X}) \mathbf{X}^{-\top} $ |
| $ \mathbf{X}^{-1} $ | $ -\mathbf{X}^{-1} \otimes \mathbf{X}^{-\top} $ |


# BLUE in Markov theorem

1. **Linear Model**:
   A linear regression model can be represented as:
   $$
   y = \mathbf{X}\beta + \epsilon
   $$
   where:
   - $ y $ is the vector of observed values.
   - $ \mathbf{X} $ is the matrix of predictor variables (features).
   - $ \beta $ is the vector of coefficients (parameters) to be estimated.
   - $ \epsilon $ is the vector of random errors, assumed to have a mean of zero.

2. **Unbiased Estimator**:
   An estimator $ \hat{\beta} $ is said to be unbiased if:
   $$
   E[\hat{\beta}] = \beta
   $$
   This means that, on average, the estimator correctly estimates the true parameter value.

3. **Best Estimator**:
   The term "best" in BLUE refers to the estimator having the smallest variance among all linear unbiased estimators. In other words, it provides the most precise estimates.

### Gauss-Markov Theorem

The Gauss-Markov theorem states that under certain assumptions, the Ordinary Least Squares (OLS) estimator $ \hat{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top y $ is the BLUE for the linear model. The assumptions are:

1. **Linearity**: The relationship between the dependent variable and the independent variables is linear.

2. **Independence**: The errors $ \epsilon $ are statistically independent.

3. **Homoscedasticity**: The variance of the errors is constant across all levels of the independent variables (no heteroscedasticity).

4. **Zero Mean of Errors**: The expected value of the errors is zero, $ E[\epsilon] = 0 $.

5. **No Perfect Multicollinearity**: The independent variables are not perfectly correlated.


# Quality measure function for regression task

### 1. **Mean Absolute Error (MAE)**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- **Description**: Measures the average absolute difference between actual and predicted values.
- **Interpretation**: Provides a straightforward interpretation of average error in the same units as the target variable.
- **Sensitivity**: Less sensitive to outliers compared to squared metrics since it uses absolute values.

### 2. **Mean Squared Error (MSE)**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- **Description**: Measures the average squared difference between actual and predicted values.
- **Interpretation**: Penalizes larger errors more than smaller ones, making it sensitive to outliers.
- **Sensitivity**: High sensitivity to outliers can lead to misleading conclusions if the data contains extreme values.

### 3. **Root Mean Squared Error (RMSE)**

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- **Description**: The square root of the MSE, bringing the error metric back to the original units of the target variable.
- **Interpretation**: Provides a more interpretable measure of average error, still sensitive to outliers.
- **Usage**: Commonly used in practice due to its ease of interpretation.

### 4. **R-squared (Coefficient of Determination)**

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

- **Description**: Represents the proportion of the variance in the dependent variable that can be explained by the independent variables in the model.
- **Interpretation**: Values range from 0 to 1; a higher value indicates a better fit. However, it can be misleading, especially in non-linear models.
- **Limitations**: Can be artificially inflated by adding more predictors, even if they don't improve model performance.

### 5. **Adjusted R-squared**

$$
R^2_{\text{adj}} = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}
$$

- **Description**: A modification of R-squared that adjusts for the number of predictors in the model.
- **Interpretation**: More reliable than R-squared for comparing models with different numbers of predictors; increases only if the new predictor improves the model more than expected by chance.
- **Usage**: Useful for model selection and comparing the quality of different regression models.

### 6. **Mean Absolute Percentage Error (MAPE)**

$$
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

- **Description**: Measures the average absolute percentage error between actual and predicted values.
- **Interpretation**: Expresses error as a percentage, making it easier to understand across different scales.
- **Limitations**: Can be undefined or misleading if any actual value $ y_i $ is zero.

### 7. **Symmetric Mean Absolute Percentage Error (SMAPE)**

$$
\text{SMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{\frac{|y_i| + |\hat{y}_i|}{2}}
$$

- **Description**: Measures the average absolute percentage error between actual and predicted values, symmetrically.
- **Interpretation**: Provides a percentage error that is normalized to the average of actual and predicted values, which helps mitigate issues that arise with values close to zero.
- **Sensitivity**: Less sensitive to extreme values than MAPE, making it more robust in cases where the actual values are small.

#### MAPE and SMAPE comparison

1. Both MAPE and sMAPE are scale-independent metrics, which means that they can be used to compare different datasets and different models.
2. MAPE might reach arbitrary big values, while sMAPE will have an upper bound (either 200 or 100, depending on the implementation).
3. Both metrics are known to assign unequal weights to overshooting and undershooting.
4. MAPE and sMAPE both don't have continuous derivatives and have issues with values being close to 0.
5. Neither MAPE nor sMAPE would make much sense when 0 is an arbitrary value, for example, with temperatures on the Fahrenheit or Celsius scales.

### Updated Comparison Summary

| Metric         | Sensitivity to Outliers | Interpretation        | Units             | Use Case                                   |
|----------------|-------------------------|-----------------------|-------------------|--------------------------------------------|
| MAE            | Low                     | Average absolute error | Same as $ y $   | General use, robust to outliers            |
| MSE            | High                    | Average squared error  | Squared units     | When larger errors are more critical       |
| RMSE           | High                    | Root of MSE           | Same as $ y $   | Commonly used, interpretable               |
| R-squared      | N/A                     | Proportion of variance | None              | Model fit, explained variability            |
| Adjusted R-squared | N/A                 | Adjusted fit measure   | None              | Comparing models with different predictors   |
| MAPE           | Moderate                | Average percentage error| Percentage        | Relative error measurement                  |
| SMAPE          | Moderate                | Symmetric percentage error| Percentage      | Better for data with small actual values    |

# Subgradient

### What is a Subgradient?

A **subgradient** is a generalization of the gradient for functions that are not differentiable. In standard optimization, gradient-based methods (like gradient descent) require a function to be differentiable so that we can compute the gradient (the slope or the rate of change) at each point. However, many important functions, especially in optimization (e.g., absolute value, max functions, hinge loss in SVMs), are not differentiable at certain points.

A **subgradient** allows us to extend the concept of gradients to non-differentiable functions. For a convex function $ f $, at a non-differentiable point $ x_0 $, a vector $ g $ is called a **subgradient** if it satisfies the following inequality for all $ x $:

$$
f(x) \geq f(x_0) + g^\top (x - x_0)
$$

This inequality essentially means that the function lies above the line defined by the subgradient $ g $ at the point $ x_0 $. In geometrical terms, this says that $ g $ is a "slope" that forms a supporting hyperplane to the function at the point $ x_0 $.

At a differentiable point, the subgradient is simply the gradient. However, at non-differentiable points, the subgradient can take a range of values (forming a set known as the **subdifferential**), reflecting the fact that the function can have multiple directions of steepest ascent or descent.

### Using Subgradients in Optimization

Subgradients can be used in optimization for functions that are not differentiable, and this is particularly useful in methods like **subgradient descent**, which is a generalization of gradient descent for non-differentiable convex functions.

#### How it works:

- For a differentiable convex function, gradient descent updates the parameter $ x $ as:
  $$
  x_{t+1} = x_t - \eta_t \nabla f(x_t)
  $$
  where $ \nabla f(x_t) $ is the gradient at $ x_t $ and $ \eta_t $ is the step size or learning rate.

- For a non-differentiable convex function, gradient descent can be generalized by replacing the gradient $ \nabla f(x_t) $ with a subgradient $ g_t $:
  $$
  x_{t+1} = x_t - \eta_t g_t
  $$
  where $ g_t $ is any subgradient at $ x_t $. Since a function can have many subgradients at a non-differentiable point, the algorithm just picks one at random (or based on some rule).


  # Condition number

The **condition number** of a matrix is a measure that describes how sensitive the solution of a system of linear equations is to changes in the input data or to perturbations in the matrix itself. It is defined as the ratio of the largest singular value of the matrix to the smallest singular value. For a matrix $ A $, the condition number $ \kappa(A) $ can be expressed as:

$$
\kappa(A) = \|A\| \cdot \|A^{-1}\|
$$

or, in terms of singular values:

$$
\kappa(A) = \frac{\sigma_{\text{max}}(A)}{\sigma_{\text{min}}(A)}
$$

where $ \sigma_{\text{max}} $ and $ \sigma_{\text{min}} $ are the largest and smallest singular values of $ A $, respectively.

### Interpretation of Condition Number

1. **Low Condition Number**:
   - If $ \kappa(A) $ is close to 1, the matrix is said to be **well-conditioned**. Small changes in the input will result in small changes in the output solution. This means the solution to the system of equations is stable and reliable.

2. **High Condition Number**:
   - If $ \kappa(A) $ is large (much greater than 1), the matrix is **ill-conditioned**. This means that small changes in the input or the matrix can result in large changes in the output solution. This instability can lead to unreliable and sensitive results.

### Correlation with Unstable Solutions in Linear Regression

In the context of linear regression, the design matrix $ \mathbf{X} $ plays a crucial role in determining the stability of the solution for the coefficients $ \beta $. When fitting a linear model, the coefficients are often calculated using the formula:

$$
\hat{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top y
$$

Here, the condition number of $ \mathbf{X}^\top \mathbf{X} $ is particularly important:

- **High Condition Number**:
  - If $ \mathbf{X}^\top \mathbf{X} $ has a high condition number, it indicates that the matrix is ill-conditioned. This can happen due to multicollinearity (when predictors are highly correlated) or if the predictors have very different scales. As a result, the computed coefficients $ \hat{\beta} $ can vary significantly with small changes in the data, leading to an unstable and unreliable model.

- **Effect on Predictions**:
  - In such cases, even slight noise in the data or measurement errors can dramatically affect the estimated coefficients and, consequently, the predictions made by the model. This can undermine the model's predictive power and interpretability.

### Practical Implications

- **Model Assessment**:
  - When evaluating a regression model, it is essential to assess the condition number of the design matrix $ \mathbf{X} $. If the condition number is high, it may warrant further investigation into the predictors and their correlations.

- **Remedial Measures**:
  - Techniques such as standardization of predictors, removing or combining highly correlated features, or using regularization methods (like Ridge or Lasso regression) can help address issues related to high condition numbers.
