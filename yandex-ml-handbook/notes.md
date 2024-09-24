# Notes from Yandex ML Handbook

## Introduction

Перед решением задачи необходимо определить к какому виду относится задача (классификация, регрессия, кластеризация...).

Далее выбираем метрику качества. Они имеют следующую иерархию:
1. Бизнес-метрики - показатели работы всей системы в целом, обычно они зависят не только от качества работы разработанной модели. Например, доход с торговой точки
2. Online-метрики - покатели работающей системы, которые высчитывают в реальном времени после внедрения модели. Например, медианное время проведенное в игре пользователем.
3. Ассесоры - показатели, оцененные специальными людьми - ассесорами (тестировщиками). Например, тестировщики оцениват качество ответа языковой модели. 
4. Offline-метрики - метрики, рассчитываемые при разработке модели, например используя исторические данные. Здесь мы используем классические метрики машинного обучения.

Также важно обратить внимание на данные, которые будут использованы при разработке модели. Являются ли они размеченными или нет, какого они вида (требуется ли feature engineering или representation learning), сколько у нас данных и какого они качества. Также отметим некоторые проболемы в данных, которые часто встречаются:
1. Пропуски
2. Выбросы
3. Ошибки разметки
4. Data drift

Далее выбираем модель и алгоритм её обучения. 

Важен и этап деплоймента модели. Необходимо эффективно запрограммировать модель и успешно встроить её в уже существующую систему. И подумать как и какие рассчитывать online-метрики. Также имеет смысл провести АБ-тестирование, то есть сравнение с предыдущей версией модели на случайно выбранных подмножествах пользователей или сессий. Если новая модель работает не очень здорово, должна быть возможность откатиться к старой.

После деплоймента модели важно продолжать дообучать или переобучать её при поступлении новых данных, а также мониторить качество. Существует concept drift — изменение зависимости между признаками и таргетом. Например, если вы делаете музыкальные рекомендации, вам нужно будет учитывать и появление новых треков, и изменение вкусов аудитории.

Data scientists обычно выбирают одни метрики для наблюдения за тренировкой модели и другой набор метрики при представлении результатов работы модели для работодателей/бизнеса и т.д. При этом выбор таких метрик должен основываться на следующих параметрах:
1. Насколько много выбросов в данных и как мы хотим их учитывать
2. Если разница между overforecating и unforecasting, и если она есть, каким образом мы будем её оценивать
3. Scale-dependent (MAE, MSE) и scale-independent (R2, NMAE). Scale-dependent изменяются при изменении величины данных, их удобно использовать для оценки общих ошибок в шкале единиц измерения таргета на каком-либо датасете, имеющим одну шкалу. Scale-independent используют нормализацию в каком-либо виде и поэтому не зависят от шкал данных, они удобны для сравнения работы модели на различных датасетах, имеющиъ различные шкалы.

При выборе метрик в задаче классификации следует опираться на следующее:
1. Распределение объектов по классам (сбалансированная выборка или нет). Для сбалансированных выборок хорошо подходит метрика accuracy, для несбалансированных presicion, recall, F1-score, area under the precision-recall curve (AUPRC)
2. Важность типа ошибок (ошибки 1-го рода или ошибки 2-го рода важнее). Исходя из важности можно смотреть на precision, recall или f1-score
3. Интерпретируемость для бизнеса
4. Цель модели. Важно ли нам создать обощенную модель (которая будет достаточно хорошо определять каждый класс) или же узко-направленную модель (которая будет хорошо определять только 1 конкретный класс). В зависимости от этого важность метрик будет различаться
5. Threshold sensibility. Метрика AUPRC не зависит от порога вероятности предсказания класса 
6. Количество классов (бинарная классификация или многоклассовая). В зависимости от этого confusion matrix, accuracy, precision, recall, f1-score могут быть адаптированы для многоклассовой классификации, используя macro, micro, weighted averages
7. Вычислительная сложность

### Common Metrics for Classification:

#### **1. Accuracy**
- **Definition**: The proportion of correctly classified instances out of the total instances.
- **Formula**:
  $$
  \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
  $$
  where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.
- **Comment**: Accuracy can be misleading for imbalanced datasets, as it does not differentiate between types of errors (false positives vs. false negatives).

#### **2. Precision (Positive Predictive Value)**
- **Definition**: The proportion of true positive predictions among all positive predictions (i.e., it measures how many of the predicted positives are actually correct).
- **Formula**:
  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $$
- **Comment**: Precision is useful when the cost of false positives is high (e.g., in spam detection, you don’t want to classify important emails as spam).

#### **3. Recall (Sensitivity or True Positive Rate)**
- **Definition**: The proportion of true positive predictions among all actual positives (i.e., how well the model identifies true positives).
- **Formula**:
  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$
- **Comment**: Recall is critical in cases where missing a positive case (false negatives) is costly (e.g., in medical diagnoses, failing to detect a disease is more serious than a false positive).

#### **4. F1-Score**
- **Definition**: The harmonic mean of Precision and Recall. It provides a balanced measure that takes both false positives and false negatives into account.
- **Formula**:
  $$
  \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$
- **Comment**: F1-Score is useful when you want a balance between precision and recall, especially in cases of class imbalance.

#### **5. AUROC (Area Under the ROC Curve)**
- **Definition**: The area under the Receiver Operating Characteristic (ROC) curve. The ROC curve plots the **True Positive Rate (Recall)** against the **False Positive Rate (FPR)** at different thresholds.
- **Formula for FPR**:
  $$
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
  $$
- **Comment**: AUROC measures the model’s ability to distinguish between classes. A value of 0.5 represents random guessing, while a value of 1 represents perfect classification.
- **Properties**:
1. Scale-invariant. It measures how well predictions
are ranked, rather than their absolute values.
2. Classiﬁcation-threshold-invariant. It measures the
quality of the model's predictions irrespective of
what classiﬁcation threshold is chosen.

#### **6. AUPRC (Area Under Precision-Recall Curve)**
- **Definition**: The area under the Precision-Recall curve. This curve plots Precision against Recall for various thresholds.
- **Comment**: AUPRC is particularly useful for **imbalanced datasets** because it focuses on the performance of the positive class, especially when the negative class is overwhelmingly larger.

#### **7. Confusion Matrix**
- **Definition**: A table that summarizes the performance of a classification model by showing the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
- **Structure**:
  |                | Predicted Positive | Predicted Negative |
  |----------------|-------------------|-------------------|
  | Actual Positive | True Positive (TP) | False Negative (FN)|
  | Actual Negative | False Positive (FP)| True Negative (TN) |
  
- **Comment**: The confusion matrix is a fundamental tool for understanding classification performance and calculating other metrics.

### **Metrics for Multi-Class Classification:**

In multi-class classification, the metrics can be calculated per class and then averaged across classes.

#### **1. Accuracy**
- **Definition**: The overall correctness, calculated as the proportion of correctly classified instances among all instances.
- **Formula**: Same as for binary classification, but counts true positives and negatives for all classes.

#### **2. Precision, Recall, and F1-Score**
- **Definition**: These metrics can be extended to multi-class classification by computing them for each class and then averaging across all classes. There are three ways to average:
  - **Macro averaging**: Average of metrics calculated independently for each class (gives equal weight to each class).
  - **Micro averaging**: Calculate the metrics globally by summing true positives, false positives, and false negatives across all classes.
  - **Weighted averaging**: Average metrics across classes, weighted by the number of instances in each class (useful for class imbalance).
  
#### **3. Confusion Matrix**
- **Definition**: In multi-class classification, the confusion matrix is extended to an $n \times n$ matrix, where $n$ is the number of classes. Each entry $C_{i,j}$ represents the number of instances of class $i$ predicted as class $j$.

#### **4. Logarithmic Loss (Log Loss)**
- **Definition**: Also known as cross-entropy loss, it measures the uncertainty of predictions by comparing the predicted probability distribution over classes with the true distribution (usually one-hot encoded).
- **Formula**:
  $$
  \text{Log Loss} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
  $$
  where $y_{i,c}$ is the true label (0 or 1), and $p_{i,c}$ is the predicted probability for class $c$ of instance $i$.

- **Comment**: Log loss is useful for probabilistic classifiers and penalizes confident but incorrect predictions more heavily than less confident ones.

#### **5. Cohen’s Kappa**
- **Definition**: Cohen’s Kappa measures the agreement between predicted and actual classifications, adjusted for the chance of random agreement.
- **Formula**:
  $$
  \kappa = \frac{p_o - p_e}{1 - p_e}
  $$
  where $p_o$ is the observed agreement (accuracy) and $p_e$ is the expected agreement by chance.
- **Comment**: Cohen’s Kappa is useful when dealing with imbalanced data and gives a more nuanced view than accuracy.

### **Precision-Recall Curves vs. ROC Curves**

- **Precision-Recall Curves**: Preferable for **imbalanced datasets**, especially when the **positive class** is of primary interest. PR curves focus on how well the model identifies the positive class while minimizing false positives.
  
- **ROC Curves**: Provide a more balanced view of performance across classes and are easier to interpret when the dataset is **balanced**. The ROC curve shows how well the model distinguishes between the positive and negative classes by plotting recall (true positive rate) against the false positive rate.

### **Choosing the Right Curve**:
- Use **Precision-Recall Curves** when the **positive class is rare** or the dataset is highly imbalanced, and you're interested in optimizing performance for the positive class.
- Use **ROC Curves** for more **balanced datasets** or when you need to assess the overall discriminative ability of the classifier across all thresholds.

### Averaging Techniques for Multi-class Classification:

When adapting binary classification metrics (like precision, recall, F1-score) to multi-class classification, we compute metrics for each class and then combine them using different averaging techniques. The most common techniques are **macro**, **micro**, **weighted**, and for multi-label problems, **samples averaging**.

#### **1. Macro Averaging**
- **Definition**: In macro averaging, the metric (e.g., precision, recall, F1-score) is calculated **independently for each class**, and then the arithmetic mean of these values is taken across all classes.
- **Formula**:
  $$
  \text{macro\_metric} = \frac{1}{N} \sum_{i=1}^{N} \text{metric}(i)
  $$
  where $N$ is the number of classes, and $\text{metric}(i)$ refers to the metric computed for class $i$.

- **When to Use**:
  - Use **macro averaging** when all classes are equally important, regardless of their frequency in the dataset. This method gives equal weight to each class, so it's useful for problems where performance on minority classes is just as critical as on majority classes.
  - **Example**: In a disease detection problem, even if one disease is rare, you might still want to evaluate its classification performance equally with common diseases.

- **Drawback**: In imbalanced datasets, macro averaging may **overemphasize the performance** on minority classes, which may have less data, potentially skewing the overall performance metric.

#### **2. Micro Averaging**
- **Definition**: In micro averaging, the true positives (TP), false positives (FP), and false negatives (FN) are aggregated **across all classes**, and then the metric is computed from these aggregated counts. It treats the problem as a single multi-class task rather than focusing on individual class metrics.
- **Formula**:
  $$
  \text{micro\_metric} = \frac{\sum \text{TP}_{i}}{\sum (\text{TP}_i + \text{FP}_i + \text{FN}_i)}
  $$
  where the sums are across all classes $i$.

- **When to Use**:
  - Use **micro averaging** when **each instance is equally important**, regardless of which class it belongs to. This method is useful when you want to optimize for the **total number of correct predictions** across all classes, rather than focusing on individual class performance.
  - **Example**: If you’re evaluating an email classification system and you care more about the total number of correct classifications rather than performance on any specific category (spam vs. non-spam), micro averaging is appropriate.

- **Strength**: Micro averaging is effective when you have **imbalanced classes** but still care about the overall performance in terms of the total number of correct predictions.

#### **3. Weighted Averaging**
- **Definition**: In weighted averaging, the metric is calculated **independently for each class**, but the final average is **weighted by the number of instances** in each class. This approach accounts for class imbalance by giving more influence to metrics from more frequent classes.
- **Formula**:
  $$
  \text{weighted\_metric} = \sum_{i=1}^{N} w_i \cdot \text{metric}(i)
  $$
  where $w_i = \frac{\text{number of instances in class } i}{\text{total number of instances}}$.

- **When to Use**:
  - Use **weighted averaging** when the dataset is **imbalanced** and you want to ensure that the overall metric reflects the performance on **more frequent classes** more heavily.
  - **Example**: In a text classification problem with several categories where some categories are much more common than others, weighted averaging will provide a more realistic overall metric by giving more importance to the classes with more data.

- **Benefit**: Weighted averaging provides a balanced view of performance that reflects the dataset’s class distribution, making it suitable for imbalanced data.

#### **4. Samples Averaging (for Multi-label Classification)**
- **Definition**: In **multi-label classification**, each instance can belong to multiple classes (labels). **Samples averaging** computes metrics (like precision, recall, or F1-score) at the **instance level**. For each instance, the metric is calculated considering **all the labels** associated with that instance, and then the average is taken across all instances.
- **Formula**:
  $$
  \text{samples\_metric} = \frac{1}{N} \sum_{i=1}^{N} \text{metric for instance } i
  $$
  where $N$ is the total number of instances.

- **When to Use**:
  - Use **samples averaging** in **multi-label classification** problems where you want to assess the performance at the instance level, considering all the labels assigned to each instance.
  - **Example**: In a multi-label image classification problem where an image can have multiple tags (like "cat," "dog," and "tree"), samples averaging gives insight into how well the model performs across all labels for each individual image.

- **Strength**: This method is especially suited for multi-label problems, where evaluating performance based on individual labels may not be enough.

### Summary of When to Use Each Averaging Method:

1. **Macro Averaging**:
   - Use when **all classes are equally important**, regardless of their frequency.
   - Best for **balanced multi-class problems** or when minority classes are of significant interest.

2. **Weighted Averaging**:
   - Use when classes are **imbalanced**, and you want to account for the frequency of each class in the overall metric.
   - Best for **imbalanced multi-class problems** where you want a metric that reflects the **distribution of classes**.

3. **Micro Averaging**:
   - Use when you care about the **total number of correctly predicted instances**, regardless of which class they belong to.
   - Best for **multi-class or multi-label problems** where **every instance is equally important**.

4. **Samples Averaging**:
   - Use specifically for **multi-label classification** when you want to evaluate performance at the level of **individual instances** rather than across labels.

## Линейные модели

Преимущество линейных моделей является их интерпретируемость, так как по значениям весов можно судить о важности признака и о влиянии его на таргет. Однако интерпретируемость может сильно снизиться, если избыточно применять feature engineering - добавив большое число сложных дополнительных фичей.

При разговоре о важности признака (на основе значения веса) необходимо учитывать его масштаб.

Функция, показывающая насколько часто наша модель ошибается называется функцией потерь, функционалом качества или loss function. От её выбора зависит то, насколько задачу в дальнейшем легко решать, и то, в каком смысле у нас получится приблизить предсказание модели к целевым значениям.

При решении задачи регресии с помощью линейной модели, используя метод наименьших квадратов (ordinary least squares), с точки зрения статичестики это соответствует гипотезе о том, что наши данные состоят из линейного "сигнала" и нормально распределенного "шума".

Функционал - это отображение, которое принимает на вход функцию и возвращает число. MSE(f, X, y) является функционалом.

При аналитическом решении линейной регрессии с MSE, получается w* = (X^TX)^-1 X^T y. Из линейной алгебры известно, что ранг X^TX и X одинаков. А значит матрица X^TX будет невырождена и, соответственно, обратимо в случае если в матрице X не будет линейно зависимых признаков. Однако зачастую в ML матрица X имеет приближенно зависимые столбцы признаков и это приводит к нестабильным решениям. В подобных случаях погрешность нахождения w* будет зависеть от квадрата числа обусловленности матрицы X, что очень плохо. Это делает полученное таким образом решение численно неустойчивым: малые возмущения y могут приводить к катастрофическим изменениям w*.

Число обусловленности (condition number) вычисляется как отношение максимального и минимального сингулярных (собственных) чисел матрицы.

Вычислительная сложность градиентного спуска O(NDS), а у аналитического решения O(N^2D + D^3), где N - число элементов выборки, D - число признаков, S - число итераций градиентного спуска. 

Стохастический градиент на каждом шаге вычисляет градиент не по всему датасету (сложность O(ND)), а по его подвыборке (батчу). 

Выборку делят на батчи путем изначального её перемешивания а затем просто берут по порядку батчи. Эпоха - это один полный проход семплера по выборке.

Шаги стохастического градиентного спуска заметно более шумные, но считать их получается значительно быстрее. В итоге они тоже сходятся к оптимальному значению из-за того, что матожидание оценки градиента на батче равно самому градиенту.

Преимуществом стохастического градиентного спуска является то, что в оперативной памяти требуется держать не всю выборку, а лишь батч.

Существует определённая терминологическая путаница, иногда стохастическим градиентным спуском называют версию алгоритма, в которой размер батча равен единице (то есть максимально шумная и быстрая версия алгоритма), а версии с бОльшим размером батча называют batch gradient descent.

Мультиколлинеарность признаков - это приближенная линейная зависимость признаков. Для того, чтобы справиться с этой проблемой, задачу обычно регуляризуют, то есть добавляют к ней дополнительное ограничение на вектор весов.

L2-регуляризация работает прекрасно и используется в большинстве случаев, но есть одна полезная особенность L1-регуляризации: её применение приводит к тому, что у признаков, которые не оказывают большого влияния на ответ, вес в результате оптимизации получается равным 0. Это позволяет удобным образом удалять признаки, слабо влияющие на таргет. Кроме того, это даёт возможность автоматически избавляться от признаков, которые участвуют в соотношениях приближённой линейной зависимости, соответственно, спасает от проблем, связанных с мультиколлинеарностью.

В задаче линейной классификации мы хотим минимизировать число ошибок предказаний класса. 

В простейшем случае можно выразить величину отступа (margin) M = y_i * (w, x_i). Число неверных классификаций - misclassification loss. Так как эта функция - кусочно-постоянная, то её нельзя оптимизировать градиентными методами (в каждой точке производная = 0). Поэтому есть смысл приближения этой функции другими гладкими функциями. 

Например: 
1. Ошибка перцептрона: F(M) = max(0, -M). Однако в случае минимизации функционала потерь основанной на данной функции потерь решение не единственно. Для таких случаев, возникает логичное желание не только найти разделяющую прямую, но и постараться провести её на одинаковом удалении от обоих классов, то есть максимизировать минимальный отступ. Это делает Hinge loss.
2. Hinge loss (SVM): F(M) = max(0, 1 - M). Итоговое положение плоскости задаётся всего несколькими обучающими примерами. Это ближайшие к плоскости правильно классифицированные объекты, которые называют опорными векторами или support vectors. Весь метод, соответственно, зовётся методом опорных векторов, или support vector machine, или сокращённо SVM.

Если решать задачу классификацию как задачу регресии, то это называется логистической регрессией. 

Тут используется вероятностная природа принадлежности объекта к одному или другому классу. Однако так как функция вероятности имеет область значений [0, 1] потребуются дополнительные модификации для применения регрессии. Из этой ситуации можно выйти так: научить линейную модель правильно предсказывать какой-то объект, связанный с вероятностью, но с диапазоном значений (-inf, +inf), и преобразовать ответы модели в вероятность. Таким объектом является logit или log odds – логарифм отношения вероятности положительного события к отрицательному $log(p/(1-p))$. 

То есть $(w, x_i) = log(p/(1-p))$, а следовательно $p = 1/(1+e^(-(w, x_i)) = \sigma((w, x_i))$.

Как теперь научиться оптимизировать $w$ так, чтобы модель как можно лучше предсказывала логиты? Нужно применить метод максимума правдоподобия для распределения Бернулли. С помощью этого мы находим требуемый функционал потерь, для которого потом рассчитываем градиент и используем градиентный метод для поиска минимума.

Предсказания такой модели будут вычисляться как $p = \sigma((w, x_i))$. Порог вероятности подбирается отдельно, для уже построенной регрессии, минимизируя нужную нам метрику на отложенной тестовой выборке. Например, сделать так, чтобы доля положительных и отрицательных классов примерно совпадала с реальной.

Отдельно заметим, что метод называется логистической регрессией, а не логистической классификацией именно потому, что предсказываем мы не классы, а вещественные числа – логиты.

Для решения задачи многоклассовой классификации линейными моделями мы сводим её к набору задач бинарной классификации. Есть два популярных метода это сделать one-vs-all и all-vs-all.

В случае one-vs-all мы обучаем K (количество классов) линейных бинарных классификаторов, каждый из которых предсказывает метку одного из классов: $b_k(x) = sign((w_k, x) + w_{0k})$. Каждый классификатор учится отделять свой класс от остальных. 

Логично, чтобы итоговый классификатор выдавал класс, соответствующий самому уверенному из бинарных алгоритмов. Уверенность можно в каком-то смысле измерить с помощью значений линейных функций: $argmax_k((w_k, x) + w_{0k})$.

Проблема данного подхода заключается в том, что каждый из классификаторов $b_k$ обучается на своей выборке, и значения линейных функций $(w_k, x) + w_{0k}$ или, проще говоря, "выходы" классификаторов могут иметь разные масштабы. Из-за этого сравнивать их будет неправильно. Нормировать вектора весов, чтобы они выдавали ответы в одной и той же шкале, не всегда может быть разумным решением: так, в случае с SVM веса перестанут являться решением задачи, поскольку нормировка изменит норму весов.

В случае all-vs-all мы обучаем $C^2_k$ классификаторов. Каждый классификатор (в случае линейных моделей) имеет вид $b_{ij}(x) = sign((w_{ij}, x) + w_{0ij}), i \neq j$.

Классификатор $a_{ij}(x)$ будем настраивать только по подвыборке $X_{ij}$, содержащей только объекты класса $i, j$. Чтобы классифицировать новый объект, подадим его на вход каждого из построенных бинарных классификаторов. Каждый из них проголосует за свой класс; в качестве ответа выберем тот класс, за который наберется больше всего голосов: $a(x) = argmax_k \sum_{i = 1}^K \sum_{i \neq j} I[a_{ij}(x) = k]$. 

Бинарную логистическую регрессию можно обобщить на многоклассовую. Допустим мы построили $K$ моделей, которые выдают логиты, которые мы потом переводим в вероятности принадлежности к какому-то одному классу. Для одновременного преобразования логитов от каждой модели в вероятности каждого класса можно использовать softmax, который производит нормировку вектора логитов. Обучать веса моделей предлагается с помощью метода максимального правдоподобия: так же, как и в случае с двухклассовой логистической регрессией.

Для увеличения масштабируемости линейных моделей по количеству фичей можно использовать разреженное кодирование, то есть вместо плотного вектора хранить словарь, в котором будут перечислены индексы и значения ненулевых элементов вектора.

Выводы:
1. На линейную модель можно смотреть как на однослойную нейросеть, поэтому многие методы, которые были изначально разработаны для них, сейчас переиспользуются в задачах глубокого обучения, а базовые подходы к регрессии, классификации и оптимизации вообще выглядят абсолютно так же. Так что несмотря на то, что в целом линейные модели на сегодня применяются редко, то, из чего они состоят и как строятся, знать очень и очень полезно.
2. Решение любой ML-задачи состоит из выбора функции потерь, параметризованного класса моделей и способа оптимизации.

## Метрические алгоритмы
Метрические практически не имеют фазы обучения, они просто запоминают выборку при тренировке (lazy learning) и на фазе инференса производят вычисления. Это непараметрические модели.

Алгоритм kNN - самый популярный метрический метод в машинном обучении (сейчас уже не применяется, но может использоваться в качестве baseline модели). Его суть в поиске наиболее близких объектов (в смысле определенной метрики) и исходя из их таргетов, вычисляется таргет объекта.
Для оценки вероятностей классов, можно рассчитать частоты классов среди ближайших объектов.