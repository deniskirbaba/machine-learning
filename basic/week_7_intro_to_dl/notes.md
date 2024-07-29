# Neural Networks
Нейронные сети могут решать практически любую задачу, однако требуется их правильно применять, понимая что они могут (почти всё) и главное чего они не могут и при каких условиях.

Обычная логистическая регрессия - это типичная нейронная сеть. У нас есть линейное преобразование и некоторое нелинейное преобразование.

Нейронки - это линейные модели "на стероидах", потому что мы даём нейронным сетям возможность самостоятельно выбирать нелинейные преобразования признаков.

Предпосылки нейронок:
1. Выбирать правильное признаковое описание для задачи ML - это дело очень сложное (feature engineering), так как нет никакого чёткого алгоритма для этого.
2. Мы хотим оптимизировать функцию потерь градиентным методом - так как это быстро и эффективно (соответственно на этом шаге "отаваливаются" деревья решений, хотя они и "автоматически" делают нелинейные преобразования, однако оптимизируются "жадным" методом)

Classic ML pipeline: X -> Feature Extractor (manual) -> Classifier (diff params $\theta_1$) -> Prediction

Хочется чтобы извлечением признаков и формированием признакового описания занимался какой-то параметрический метод, по которому можно найти оптимальные параметры градиентным методом (как и в Classifier).

NN pipeline: X -> Feature Extractor (diff params $\theta_2$) -> Classifier (diff params $\theta_1$) -> Prediction

При Blending-е у нас в качестве Feature Extractor-ов выступают различные модели, которые выдают нам нелинейные преобразования признаков так, чтобы потом линейная модель Classifier могла с ними справиться.

Однако в таком случае, у нас Feature Extractor и Classifier будут разделены и обучаться отдельно, а мы хотим связть их и сделать чтобы они были дифференцируемы и обучались градинетным методом одновременно. Например:

NN example pipeline: X -> Feature Extractor (Linear model + Sigmoid) -> Logistic Regression -> Prediction

То есть мы одновременно обучаем и Feature Extractor и Logistic Regression.

Если подытожить, то вместо прямого преобразования из пространства признаков в пространство ответов - мы последовательно будем преобразовывать признаковые пространства (получая некоторые промежуточные пространства) и в конце получаем предсказания.
Этот подход логичен, так как решить задачу сходу может быть очень сложно.

Нейронные сети - последовательность дифференцируемых линейных и нелинейных преобразований. 

Функции активации:
1. Sigmoid(a) = 1/(1 + e^(-a))
2. Гиперболический тангенс(a) = tanh(a)
3. ReLu(a) = max(0, a)
4. Softplus(a) = log(1 + e^a)

# Термины
1. Слой - преобразование признакового пространства, осуществляющее переход от одного признакового пространства в другое. Бывают различных типов: полносвязный слов (dense layer) ($Wx+b$), может задаваться нелинейной функцией, например сигмоида. К слою могут быть прикреплено нелинейное преобразование - в общем может быть как угодно. Не важно сколько и каких слоёв в нейронке, важно лишь то, насколько нейронка способна решать определенные задачи и какие она имеет структурные особенности.
2. Функция активации - они как раз определяют нелинейности в слоях. Они применяются к выходу слоя. ФУНКЦИИ АКТИВАЦИИ ПРИМЕНЯЮТСЯ ПОЭЛЕМЕНТНО. 
3. Backpropogation - алгоритм обучения нейронной сети - метод взятия производной сложной функции.

# Backpropogation
Каждую функцию мы можем представить в виде графа вычислений. Тогда каждым листом будет переменная, а каждый промежуточный узел - какая-то операция. 
При вычислении значения функции мы идём в одну сторону.
Тогда, имея такое представления для расчета производной функции по каждой переменной - будем идти в обратном порядке и находить производные. 
Вычисленные таким образом градиенты затем применяются для обновления весов параметров каждого слоя.
Вообще говоря, для каждого графа вычислений строится другой граф - вычисления производной, который затем используется для вычисления градиентов (то етсь не нужно на каждой итерации заново вычислять производные). 

Под "капотом" для вычисления значения выхода нейронной сети по входу (forward pass) создаётся граф вычислений (направленный ациклический граф - directed acyclic graph), а также создаётся граф для вычисления градиентов для каждого параметра нейронной сети (backward pass).

Так как локальная производная операции "+" = 1. То этот узел просто-напросто позволяет градиенту продвигаться дальше по каждой ветви графа без изменения его значения.

Backpropagation it is just recursevily application of the chain rule backwards through the computational graph.

# Activation functions
1. Сигмоида. Область значений (0, 1) смещена относительно нуля (а в нейронках (тут много линейных моделей) требуется нормировать данные), дорого считать экспоненту.
2. Гиперболический тангенс - несмещена (область значений -1, 1), но так же дорого сичать экспоненты
3. ReLU - rectified linear unit. max(0, a). Легко вычислять (как значение так и производную) (быстрее до 6 раз чем сигмоида). Выход не центрированный. Проблема нулевых градиентов при x<0 (а значит градиент не пойдет дальше аргументов, на которых x < 0). При использовании ReLU требуется вставлять много нормировочных слоев, так как у на выход нецентрированный.
4. Leaky ReLU - справа от нуля так же как и ReLU, а слева по-другому. max(0.001a, a). Убирает проблему нулевых градиентов при x<0.
5. ELU - справа от нуля также как и ReLU, а слева - экспоненциальная фуникця. Но не особо используется, так как требуется считаьт экспоненту
6. GELU - Gaussian error linear unit. Работет лучше ReLU, ELU. GELU ведет себя асимптотически также как и ReLU, однако различается около нуля. $GELU(a) = a P(X<=a)=x Ф(a), Ф(а) - функция вероятноссти нормального распределения$. 
7. SiLU - sigmoid-weighted linear unit - похож на GELU. $SiLU(a) = a*\sigma(a)$. Использовался в RL

По факту функция активации влияет на скорость обучения. И малые измененения в них могут приводить лишь к изменению скорости обучения.

Мы хотим чтобы функции активации, как и данные были центрированы (нормированны), так как при работе с линейными моделями если у нас все данные будут положительны при подаче на вход функции активации, то и с нее будет выход линейный, тогда получится что в нашей модели не будет нелинейного преобразования.

Как выбирать функцию активации:
1. Если нет никаких ограничений (в том числе на область значений выходов функции активации) - использовать ReLU и нормировать данные (не только в начале но и в середине сетки)
2. Функция активации - это по факту гиперпараметр. Однако в нейронных сетях такое большое количество гиперпараметров, что их выбор происходит полу-интуитивно
3. Можно попробовать использовать Leaky ReLU, ELU, GELU, SiLU
4. tanh используется там где есть ограничения на область значений
5. сигмоида используется крайне редко, только если у нас там где-то есть бинарная классификация

# Проблема затухающих градиентов (Vanishing gradient)
Так как градиент - композиция частных производных на определенных слоях, то в этих частных производных есть компоненты производных от функций активаций. Сигмоида плоха тем, что имеет области насыщения (в которых производная будет 0), а также имеет максимальное значение производной =0.25. Соответственно каждый раз когда мы ставим сигмоиду в нейронную сеть у нас градиент будет уменьшаться минимум в 4 раза. У ReLU затухание градиента будет только с одной стороны.

Проблему затухающих градиентов нельзя решить простым домножением градиента на большое число. При домножении величины (градиент) на малую величину у нас уменьшается как сам сигнал так и присутствующий в нем шум, однако при умножении на малую величину у нас разница в значениях между сигналом и шумом уменьшается, а следовательно при дальнейшем умножении мы получим очень зашумленное значение.

Есть и другая проблема - взрыв градиентов. Она происзодит когда мы получаем большие частные производные из-за этого получаем большой градиент по весам.
Эту проблему решают "сжиманием" веткора градиента до какого-то оптимального значения (Gradient Clipping).


# Gradient Optimization
Определение learning rate - это тоже поиск гиперпараметра. Его нужно выбирать по поведению функции потерь на обучении (в среднем должно выглядеть как перевернутая функция логарифма). 

Существует большое количество оптимизацторов: Momentum, AdaGRad, Adadelta, RMSprop, Adam...

## SGD
При использовании стандартного SGD у нас градиент считается по батчу. Если же у нас размер батча достаточно мал, то получается что у нас изменение градиента будет шумным (однако это не особо проблема, так как шум у нас центрирован и в среднем градиент будет идти куда надо).

## SGD with momentum
Momentum: переиспользование градиентов с предыдущих шагов
На каждом шаге будет учитывать не только данный градиент, но и усредненные градиенты с предыдущих шагов

## SGD with Nesterov momentum
Nesterov Momentum: мы вначале шагаем вдоль накопленного градиента с прошлых шагов, а затем считаем градиент в той точке в которую попали и шагаем вдоль него

Однако и в momentum и в Nesterov momentum у нас будет бОльший расход памяти, так как нам требуется хранить еще одну величину (усредненных градиентов с прошлых шагов) для каждых параметрах во всех слоях.

## AdaGrad (Adaptive gradients)
https://machinelearningmastery.com/gradient-descent-with-adagrad-from-scratch/
A limitation of gradient descent is that it uses the same step size (learning rate) for each input variable. This can be a problem on objective functions that have different amounts of curvature in different dimensions, and in turn, may require a different sized step to a new point.

Adaptive Gradients, or AdaGrad for short, is an extension of the gradient descent optimization algorithm that allows the step size in each dimension used by the optimization algorithm to be automatically adapted based on the gradients seen for the variable (partial derivatives) seen over the course of the search.

The parameters with the largest partial derivative of the loss have a correspondingly rapid decrease in their learning rate, while parameters with small partial derivatives have a relatively small decrease in their learning rate.

A problem with the gradient descent algorithm is that the step size (learning rate) is the same for each variable or dimension in the search space. It is possible that better performance can be achieved using a step size that is tailored to each variable, allowing larger movements in dimensions with a consistently steep gradient and smaller movements in dimensions with less steep gradients.

AdaGrad is designed to specifically explore the idea of automatically tailoring the step size for each dimension in the search space.

This is achieved by first calculating a step size for a given dimension, then using the calculated step size to make a movement in that dimension using the partial derivative. This process is then repeated for each dimension in the search space.

Algorithm:
$$s_{i+1} = s_i + (grad_{\omega_i} L)^2$$
$$\eta_{i+1} = \frac{\eta_{initial}}{\sqrt{s_{i+1} + EPS}}$$
$$\omega_{i, t+1} = \omega{i,t} - \eta_{i+1} * grad_{\omega_i} L$$

## RMSprop (Root Mean Squared Propagation)
https://deepai.org/machine-learning-glossary-and-terms/rmsprop#:~:text=RMSProp%2C%20which%20stands%20for%20Root,in%20training%20deep%20neural%20networks.
It is, in fact, an extension of gradient descent and the popular AdaGrad algorithm and is designed to dramatically reduce the amount of computational effort used in training neural networks.

The algorithm works by exponentially decaying the learning rate every time the squared gradient is less than a certain threshold.

The use of the root mean square in RMSprop, as we will see, helps to avoid the problem of the learning rate being too small or too large. If the gradients are small, the learning rate will be increased to speed up convergence, and if the gradients are large, the learning rate will be decreased to avoid overshooting the minimum of the loss function.

RMSProp addresses the issue of a global learning rate by maintaining a moving average of the squares of gradients for each weight and dividing the learning rate by this average. This ensures that the learning rate is adapted for each weight in the model, allowing for more nuanced updates. The general idea is to dampen the oscillations in directions with steep gradients while allowing for faster movement in flat regions of the loss landscape.

The RMSProp update adjusts the Adagrad method to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, RMSProp uses an exponential decay that discards history from the extreme past so that it can converge rapidly after finding a convex bowl, as if it were an Adagrad with a fresh start.

Algorithm:
$$\omega_{t+1} = \omega_t - \eta \frac{grad_{\omega} L}{\sqrt{S_t}}$$
$$S_{t+1} = \beta S_t + (1 - \beta) (grad_{\omega} L)^2$$

## Adam
https://arxiv.org/pdf/1412.6980
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
https://www.geeksforgeeks.org/adam-optimizer/
The name Adam is derived from adaptive moment estimation.

The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

The authors describe Adam as combining the advantages of two other extensions of stochastic gradient descent. Specifically:
1. Adaptive Gradient Algorithm (AdaGrad) that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).
2. Root Mean Square Propagation (RMSProp) that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).

Adam realizes the benefits of both AdaGrad and RMSProp.

Instead of adapting the parameter learning rates based on the average first moment (the mean) as in RMSProp, Adam also makes use of the average of the second moments of the gradients (the uncentered variance).

Specifically, the algorithm calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages.

The initial value of the moving averages and beta1 and beta2 values close to 1.0 (recommended) result in a bias of moment estimates towards zero. This bias is overcome by first calculating the biased estimates before then calculating bias-corrected estimates.

In practice Adam is currently recommended as the default algorithm to use, and often works slightly better than RMSProp. However, it is often also worth trying SGD+Nesterov Momentum as an alternative.

Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances. […] its bias-correction helps Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Insofar, Adam might be the best overall choice.

