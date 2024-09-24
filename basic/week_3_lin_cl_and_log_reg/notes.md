Эмпирический риск = сумме функции потерь по каждому элементу выборки.

Balanced accuracy рассчитывает accuracy по каждому классу отдельно, а затем берет среднее по всем классам.

Логистическая регрессия задаёт нелинейное отображение, так как она задаёт отображение из пространства признаков в вероятностное пространство. Однако разделяющей поверхностью является гиперплоскость!
By using Logistic Regression we generate a Bernoulli distribution in each point of space!

# Loss Functions for Classification

Loss functions in classification tasks can be framed in two broad ways: **relative to the margin** (a distance-based approach) or **relative to probability** (a likelihood-based approach).

### 1. **Loss Functions Relative to Probability (Log-Likelihood-Based)**

These loss functions measure the gap between the predicted probabilities and the true class labels. They are often used when models output probability distributions, such as in softmax classifiers or logistic regression.

#### **Cross-Entropy Loss (Log Loss)**
- **Definition**: This is the most common loss function for classification tasks where the output is a probability distribution. Cross-entropy compares the predicted probability distribution over classes with the true distribution (which is typically one-hot encoded).
- **Formula** (for binary classification):
  $$
  \text{Loss} = -\left[ y \log(p) + (1 - y) \log(1 - p) \right]
  $$
  where $y$ is the true label (0 or 1) and $p$ is the predicted probability for the positive class.
  
  For multi-class classification, the formula generalizes as:
  $$
  \text{Loss} = -\sum_{i=1}^C y_i \log(p_i)
  $$
  where $y_i$ is the true probability for class $i$ (usually 0 or 1), and $p_i$ is the predicted probability for class $i$.

- **Intuition**: It penalizes incorrect predictions by assigning a high cost when predicted probabilities are far from the true label's probability.

#### **KL Divergence (Kullback-Leibler Divergence)**
- **Definition**: KL divergence is a measure of how one probability distribution diverges from another. It's used in classification problems that deal with probabilistic output distributions, such as when you compare the predicted probability distribution with a target distribution.
- **Formula**:
  $$
  D_{\text{KL}}(P || Q) = \sum_{i=1}^C P(i) \log\left(\frac{P(i)}{Q(i)}\right)
  $$
  where $P$ is the true distribution and $Q$ is the predicted distribution.
  
- **Intuition**: Minimizing KL divergence aligns the predicted probability distribution with the true distribution.

---

### 2. **Loss Functions Relative to Margin (Distance-Based)**

Margin-based loss functions focus on the decision boundary and the **margin** between correctly classified and misclassified points. They are more common in linear classifiers like support vector machines (SVM).

#### **Hinge Loss (Used in SVM)**
- **Definition**: Hinge loss is used primarily in SVMs. It aims to increase the margin between the decision boundary and the data points. It encourages correct classification by penalizing misclassified points or points within the margin.
- **Formula** (for binary classification):
  $$
  \text{Loss} = \max(0, 1 - y \cdot f(x))
  $$
  where $y \in \{-1, +1\}$ is the true label and $f(x)$ is the raw model output (logit or distance from decision boundary).

- **Intuition**: The loss is zero when the point is classified correctly with a margin of at least 1. If the point is misclassified or within the margin, the loss increases.

#### **Squared Hinge Loss**
- **Definition**: A variation of the hinge loss that squares the hinge loss value to penalize larger margin violations more strongly.
- **Formula**:
  $$
  \text{Loss} = \max(0, 1 - y \cdot f(x))^2
  $$
- **Intuition**: Similar to hinge loss but with a stronger penalty for points that are incorrectly classified or fall close to the margin.

---

### 3. **Softmax Margin Loss**
This loss is often seen as bridging the two paradigms. It has margin properties but operates on probabilities (similar to cross-entropy).

- **Definition**: It’s commonly used in multi-class classification tasks. The softmax function converts logits into probabilities, and then the cross-entropy between the true label and the predicted probabilities is computed. However, the raw model output (logits) is also linked to the margin concept.
- **Formula**:
  $$
  \text{Loss} = -\log \left( \frac{\exp(f(x)_y)}{\sum_j \exp(f(x)_j)} \right)
  $$
  where $f(x)_y$ is the logit corresponding to the correct class $y$, and $f(x)_j$ are the logits for all classes.

- **Intuition**: The loss function pushes the model to output a high score for the correct class while reducing the scores for incorrect classes, ensuring a clear margin in the logit space.

---

### Summary of Differences

- **Relative to Probability**: Cross-entropy and KL divergence operate by comparing predicted **probabilities** to the true class labels, focusing on how confident the model is in its predictions.
- **Relative to Margin**: Hinge loss and squared hinge loss measure how far a prediction is from the decision boundary, emphasizing classification correctness and the **distance** from the boundary.

Both types of loss functions encourage good classification, but probability-based losses are better suited for probabilistic models (e.g., neural networks), while margin-based losses are more relevant for margin-based classifiers (e.g., SVM).