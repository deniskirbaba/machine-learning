# Нейронные сети

Нейронные сети способны решать широкий спектр задач, однако важно правильно понимать их возможности и ограничения. Они могут быть очень мощными инструментами, но требуют грамотного применения и настройки, особенно в сложных задачах.

## Логистическая регрессия как нейронная сеть

Логистическая регрессия — это простейший пример нейронной сети. Она представляет собой однослойную сеть, где происходит линейное преобразование входных данных с последующим применением нелинейной активационной функции (сигмоида). 

## Нейронные сети: общая концепция

Нейронные сети можно воспринимать как расширение линейных моделей, «усиленные» возможностью обучать сложные нелинейные зависимости. Главная идея в том, что нейронные сети могут **автоматически** выбирать и обучать нелинейные преобразования признаков для решения задачи.

## Основные предпосылки для нейронных сетей:

1. **Трудности feature engineering**: Подбор правильных признаков — сложная задача. Нет универсального алгоритма для создания идеального признакового описания задачи. Нейронные сети способны решать эту проблему, обучая представления признаков автоматически.
2. **Градиентные методы оптимизации**: Нейронные сети можно эффективно обучать с помощью градиентного спуска, что позволяет быстро минимизировать функцию потерь. Это ограничивает использование деревьев решений, которые оптимизируются «жадными» методами и не могут быть интегрированы в градиентные схемы.

## Классический ML pipeline:
$$
X \rightarrow \text{Feature Extractor (manual)} \rightarrow \text{Classifier (с параметрами $\theta_1$)} \rightarrow \text{Prediction}
$$
Этот процесс требует ручного извлечения признаков, что может быть затруднительно.

## NN pipeline (нейронные сети):
$$
X \rightarrow \text{Feature Extractor (с параметрами $\theta_2$)} \rightarrow \text{Classifier (с параметрами $\theta_1$)} \rightarrow \text{Prediction}
$$
Здесь извлечение признаков и классификация объединяются в один процесс, обучаемый с помощью градиентного спуска.

## Blending и нейронные сети

При **blending** модели разных типов могут использоваться для преобразования признаков, чтобы линейная модель могла работать с нелинейными зависимостями. Однако в таком случае признаки и классификатор будут обучаться отдельно.

Нейронные сети же позволяют связать **Feature Extractor** и **Classifier**, и обучать их одновременно, используя дифференцируемые преобразования и градиентные методы. Это дает возможность оптимизировать все параметры сразу, улучшая общее качество модели.

## Последовательное преобразование признаков

Вместо прямого преобразования входных данных в предсказания, как это происходит в линейных моделях, нейронные сети последовательно преобразуют пространство признаков, создавая промежуточные представления. Это логичный подход, так как задача может быть слишком сложной для прямого решения. Последовательные преобразования позволяют постепенно выявлять сложные зависимости.

## Нейронные сети как последовательность преобразований

Нейронные сети можно описать как **последовательность дифференцируемых линейных и нелинейных преобразований**. Это позволяет обучать их с помощью градиентного спуска.

## Функции активации

Функции активации играют ключевую роль в нейронных сетях, добавляя нелинейность, которая необходима для решения сложных задач:

1. **Сигмоида (Sigmoid)**: 
   $$
   \sigma(a) = \frac{1}{1 + e^{-a}}
   $$
   Используется в логистической регрессии и на выходных слоях для бинарной классификации. Однако у нее есть недостаток — затухающий градиент при больших значениях входного сигнала.

2. **Гиперболический тангенс (Tanh)**: 
   $$
   \tanh(a) = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}}
   $$
   Преобразует входные значения в диапазон от -1 до 1. В отличие от сигмоиды, имеет центральную симметрию относительно нуля, что может ускорить сходимость градиентного спуска.

3. **ReLU (Rectified Linear Unit)**: 
   $$
   \text{ReLU}(a) = \max(0, a)
   $$
   Является одной из самых популярных функций активации благодаря своей простоте и способности справляться с проблемой затухающих градиентов. Однако у неё есть недостаток — она может «выключать» нейроны при отрицательных значениях, что приводит к мертвым нейронам.

4. **Softplus**: 
   $$
   \text{Softplus}(a) = \log(1 + e^{a})
   $$
   Является сглаженной версией ReLU, избегает проблемы «мертвых нейронов», но вычислительно более затратна.

# Термины

1. Слой - преобразование признакового пространства, осуществляющее переход от одного признакового пространства в другое. Бывают различных типов: полносвязный слой (dense layer) ($Wx+b$), может задаваться нелинейной функцией, например сигмоида. К слою могут быть прикреплено нелинейное преобразование - в общем может быть как угодно. Не важно сколько и каких слоёв в нейронке, важно лишь то, насколько нейронка способна решать определенные задачи и какие она имеет структурные особенности.
2. Функция активации - они как раз определяют нелинейности в слоях. Они применяются к выходу слоя. ФУНКЦИИ АКТИВАЦИИ ПРИМЕНЯЮТСЯ ПОЭЛЕМЕНТНО. 
3. Backpropogation - алгоритм обучения нейронной сети - метод взятия производной сложной функции.

# Backpropagation

Backpropagation (обратное распространение ошибки) — это метод, используемый для вычисления градиентов в нейронных сетях с целью оптимизации весов. Его основная идея заключается в том, что каждый параметр сети обновляется на основе того, как изменение этого параметра влияет на ошибку модели.

## Граф вычислений

Любую функцию можно представить в виде **графа вычислений**. В этом графе:
- **Листовые узлы** — это входные переменные (например, параметры модели, данные),
- **Промежуточные узлы** — это результаты операций над этими переменными (например, сложение, умножение, применение функций активации).

Когда мы выполняем прямое вычисление (forward pass), мы идем по графу от входов к выходам, вычисляя значение функции. Однако для того чтобы обучать нейронную сеть, нам нужно найти производные (градиенты) выходной функции по каждому параметру, то есть выполнить **обратный проход** (backward pass).

## Принцип обратного распространения

Для того чтобы найти производную целевой функции по каждому параметру сети, мы применяем **правило цепочки** (chain rule) в обратном порядке по графу вычислений:
- **Forward pass**: мы вычисляем значение функции, проходя через все узлы от входа до выхода.
- **Backward pass**: мы идем в обратном направлении, начиная с выхода и вычисляя производные по каждому параметру, используя правило цепочки.

Градиенты, вычисленные с помощью backpropagation, затем используются для обновления весов параметров с помощью оптимизационного алгоритма, например, **градиентного спуска**.

## Два графа: вычисление функции и градиентов

- **Прямой проход (Forward pass)**: создает **направленный ациклический граф** (directed acyclic graph, DAG), который представляет последовательность вычислений, необходимых для получения конечного результата нейронной сети.
- **Обратный проход (Backward pass)**: создает граф для вычисления производных функции по параметрам сети. Этот граф используется для эффективного вычисления градиентов без необходимости повторного символического дифференцирования на каждой итерации.

## Локальные производные и передача градиентов

Пример локальной производной:
- **Для операции сложения (+)** локальная производная равна 1. Это означает, что градиент, поступающий на узел сложения, просто передается по каждой ветви графа дальше без изменений. 

# Activation functions

1. Сигмоида. Область значений (0, 1) смещена относительно нуля (а в нейронках (тут много линейных моделей) требуется нормировать данные), дорого считать экспоненту.

2. Гиперболический тангенс - несмещена (область значений -1, 1), но так же дорого считать экспоненты

3. ReLU - rectified linear unit. max(0, a). Легко вычислять (как значение так и производную) (быстрее до 6 раз чем сигмоида). Выход не центрированный. Проблема нулевых градиентов при x<0 (а значит градиент не пойдет дальше аргументов, на которых x < 0). При использовании ReLU требуется вставлять много нормировочных слоев, так как у на выход нецентрированный.

4. Leaky ReLU - справа от нуля так же как и ReLU, а слева по-другому. max(0.001a, a). Убирает проблему нулевых градиентов при x<0

5. ELU - справа от нуля также как и ReLU, а слева - экспоненциальная фуникця. Но не особо используется, так как требуется считать экспоненту

6. GELU - Gaussian error linear unit. Работет лучше ReLU, ELU. GELU ведет себя асимптотически также как и ReLU, однако различается около нуля. $GELU(a) = a P(X<=a)=x Ф(a), Ф(а)$ - cdf нормального распределения

7. SiLU - sigmoid-weighted linear unit - похож на GELU. $SiLU(a) = a*\sigma(a)$. Использовался в RL

По факту функция активации влияет на скорость обучения. И малые измененения в них могут приводить лишь к изменению скорости обучения.

Мы хотим чтобы функции активации, как и данные были центрированы (нормированны), так как при работе с линейными моделями если у нас все данные будут положительны при подаче на вход функции активации (например ReLU), то и с нее будет выход линейный, тогда получится что в нашей модели не будет нелинейного преобразования.

Как выбирать функцию активации:
1. Если нет никаких ограничений (в том числе на область значений выходов функции активации) - использовать ReLU и нормировать данные (не только в начале но и в середине сетки)

2. Функция активации - это по факту гиперпараметр. Однако в нейронных сетях такое большое количество гиперпараметров, что их выбор происходит полу-интуитивно

3. Можно попробовать использовать Leaky ReLU, ELU, GELU, SiLU

4. tanh используется там где есть ограничения на область значений

5. сигмоида используется крайне редко, только если у нас там где-то есть бинарная классификация (вставляют как функцию активации последнего слоя)

# Проблема затухающих градиентов (Vanishing Gradient)

Проблема затухающих градиентов возникает в нейронных сетях, когда во время обратного распространения ошибки (backpropagation) значения градиентов становятся очень маленькими. Это мешает корректному обновлению весов и препятствует обучению глубоких слоев.

## Причины затухающих градиентов

Градиенты — это композиция частных производных функций активаций, применяемых на каждом слое сети. Некоторые функции активаций, такие как **сигмоида** или **гиперболический тангенс**, имеют области насыщения, где их производные очень малы или даже равны нулю:
- **Сигмоида**: производная функции имеет максимальное значение 0.25, и в области насыщения (где значения входов сильно отрицательные или сильно положительные) производная стремится к 0.
- Когда сигмоида используется на каждом слое, каждый раз градиент уменьшается минимум в 4 раза, что приводит к экспоненциальному уменьшению градиентов при распространении через слои.
- **ReLU (Rectified Linear Unit)**: затухание градиента наблюдается только с одной стороны (для отрицательных значений входа, где производная равна 0), что делает ReLU более устойчивой к затуханию градиентов по сравнению с сигмоидой, но все же оставляет проблему для отрицательных входов.

## Почему нельзя решить проблему просто увеличением градиента?

На первый взгляд, проблему затухающих градиентов можно попытаться решить, просто умножив градиенты на большое число. Однако это неэффективно по нескольким причинам:
1. **Затухание сигнала и шума**: когда мы уменьшаем значения градиентов, затухает не только сам сигнал, но и присутствующий в нем шум. Однако при масштабировании градиентов разница между сигналом и шумом уменьшается. Это приводит к тому, что дальнейшее увеличение приводит к усилению шума, а не полезного сигнала.
2. **Переобучение или зашумленность**: при дальнейших умножениях искаженный сигнал становится менее полезным для обновления параметров.

## Решения проблемы затухающих градиентов

1. **Использование других функций активации**:
   - **ReLU** и его модификации (например, **Leaky ReLU**, **ELU**) помогают снизить проблему затухания градиентов, так как их производные для положительных значений входа остаются постоянными и ненулевыми.
   - **Batch Normalization**: помогает стабилизировать и ускорить обучение, уменьшив эффект насыщения за счет нормализации активаций.

2. **Xavier и He инициализация**: специальные методы инициализации весов, такие как **Xavier initialization** и **He initialization**, помогают сделать так, чтобы градиенты не затухали и не взрывались на старте обучения.

# Проблема взрыва градиентов (Exploding Gradient)

Проблема взрыва градиентов возникает, когда во время обратного распространения ошибки значения градиентов становятся слишком большими. Это приводит к нестабильным весам и затрудняет обучение.

## Причины взрыва градиентов

Взрыв градиентов обычно происходит, когда частные производные становятся слишком большими на нескольких слоях, что приводит к экспоненциальному увеличению значения градиентов по мере их распространения через слои. Это особенно часто случается в глубоких нейронных сетях или рекуррентных нейронных сетях (RNN), где длинные цепочки умножений частных производных могут приводить к очень большим значениям градиентов.

## Решения проблемы взрыва градиентов

1. **Gradient Clipping**: метод, который применяется для "сжатия" значений градиентов, если они превышают некоторый порог. Это позволяет контролировать величину градиентов, предотвращая их чрезмерное увеличение. 
   - Идея заключается в том, что если норма градиента превышает заданное значение, то градиент масштабируется так, чтобы его норма была равна этому значению.

2. **Инициализация весов**: как и в случае с затухающими градиентами, правильная инициализация весов (например, с использованием He или Xavier инициализации) помогает снизить риск взрыва градиентов.

# Gradient Optimization

Определение learning rate - это тоже поиск гиперпараметра. Его нужно выбирать по поведению функции потерь на обучении (в среднем должно выглядеть как перевернутая функция логарифма). 

При обучении нейронных сетей ключевую роль играют оптимизаторы, которые определяют, как будут обновляться параметры модели на основе градиентов. Различные оптимизаторы предлагают разные подходы к тому, как использовать градиенты для минимизации функции потерь.

Существует большое количество оптимизаторов. Основные:
* SGD
* SGD with Momentum
* AdaGrad
* Adadelta
* RMSprop
* Adam

## Subgradient

[Wiki](https://en.wikipedia.org/wiki/Subderivative#The_subgradient)

Субградиент - обобщение градиента. Т.е. субградиент может быть определен для функции, которые не является дифференцируемой в некоторых точках, тогда получается что в этих точках существует такой набор векторов, в котором каждый вектор будет удовлетворять уравнению: 
$$f(x) - f(x_0) \ge v \cdot (x - x_0)$$

Такие вектора называются субградиентами в точке $x_0$. 

## Stochastic Gradient Descent (SGD)

**Stochastic Gradient Descent** (SGD) — это базовый метод оптимизации, в котором обновление параметров происходит на основе градиента, вычисленного по случайному батчу данных.

Основная формула обновления для параметров $\theta$:
$$ \theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta}L(\theta) $$
где $\eta$ — это скорость обучения (learning rate), а $\nabla_{\theta}L(\theta)$ — градиент функции потерь по параметрам.

### Шум в SGD:
- Если размер батча мал, градиенты могут быть шумными. Однако этот шум центрирован, что в среднем приводит к правильному направлению обновления параметров.
- Шум в SGD помогает избежать локальных минимумов, улучшая обобщающую способность модели.

## SGD with Momentum

**SGD с моментом (Momentum)** помогает сгладить процесс обучения за счет использования информации о предыдущих шагах. Это делает обучение более стабильным и ускоряет его.

Формула обновления с моментом:
$$ v_t = \gamma v_{t-1} + \nabla_{\theta}L(\theta) $$
$$ \theta_{t+1} = \theta_t - \eta v_t $$

Где $v_t$ — это накопленный градиент (velocity), а $\gamma$ — коэффициент, определяющий "инерцию" градиентов с прошлых шагов.

### Преимущества:
- Позволяет уменьшить влияние шума на обновления.
- Ускоряет обучение, особенно в задачах с высокой размерностью параметров.

### Недостатки:
- Требуется дополнительная память для хранения предыдущих градиентов.

## Nesterov Accelerated Gradient (Nesterov Momentum)

**Nesterov Momentum** — это улучшенная версия momentum. Отличие в том, что градиент вычисляется не в текущей точке, а после предварительного шага вдоль накопленного градиента.

Формула:
   $$ v_{t+1} = \gamma v_t + \nabla_{\theta}L(\theta_t + \gamma v_t) $$
   $$ \theta_{t+1} = \theta_t - \eta v_{t+1} $$

### Преимущества:
- Более точное обновление параметров за счет того, что градиент вычисляется после движения вдоль инерции.

### Недостатки:
- Требуется дополнительная память для хранения предыдущих градиентов.

## AdaGrad (Adaptive Gradient)

[Good material](https://optimization.cbe.cornell.edu/index.php?title=AdaGrad)

**AdaGrad** — это адаптивный оптимизатор, который подстраивает скорость обучения для каждого параметра в зависимости от того, как часто он обновляется. Параметры, которые изменяются часто, получают меньшие шаги обновления, а редкие параметры — большие.

Формула обновления:
$$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_{\theta}L(\theta) $$

Где $G_t$ — это диагональная матрица, представляющая сумму квадратов градиентов по каждому параметру.

### Преимущества:
- Хорошо работает в задачах с разреженными признаками.
- While standard sub-gradient methods use update rules with step-sizes that ignore the information from the past observations, AdaGrad adapts the learning rate for each parameter individually using the sequence of gradient estimates.

### Недостатки:
- Со временем шаг обучения для часто изменяемых параметров становится слишком малым, что может замедлить обучение.

## RMSprop (Root Mean Squared Propagation)

[Good material](https://optimization.cbe.cornell.edu/index.php?title=RMSProp)

**RMSprop** улучшает AdaGrad, вводя экспоненциальное скользящее среднее для квадратов градиентов. Это позволяет избежать слишком агрессивного уменьшения скорости обучения, сохраняя адаптивность.

Алгоритм:
1. Экспоненциальное сглаживание квадратов градиентов:
   $$ S_{t+1} = \beta S_t + (1 - \beta) (\nabla_{\theta} L)^2 $$
2. Обновление параметров:
   $$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{S_{t+1} + \epsilon}} \nabla_{\theta} L $$

Где $\beta$ — коэффициент затухания, обычно близкий к 0.9.

### Преимущества:
- RMSprop помогает обучению оставаться эффективным даже при больших и малых градиентах.

## Adam (Adaptive Moment Estimation)

[Good material](https://optimization.cbe.cornell.edu/index.php?title=Adam)

**Adam** объединяет идеи **Momentum** и **RMSprop**, используя экспоненциальное скользящее среднее для как градиентов, так и квадратов градиентов.

Алгоритм:
1. Экспоненциальное среднее градиентов:
   $$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{\theta} L $$
2. Экспоненциальное среднее квадратов градиентов:
   $$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_{\theta} L)^2 $$
3. Коррекция смещения для моментов:
   $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
4. Обновление параметров:
   $$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t $$

### Преимущества:
- Быстрое и стабильное обучение даже на больших и шумных данных.
- Хорошо подходит для задач с большими объемами данных и высокой размерностью.

### Недостатки:
- Параметры $\beta_1$ и $\beta_2$ требуют тщательной настройки, хотя существуют рекомендованные значения (обычно $\beta_1 = 0.9$ и $\beta_2 = 0.999$).
