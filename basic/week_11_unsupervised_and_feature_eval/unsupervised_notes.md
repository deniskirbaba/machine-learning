# Unsupervised Learning

**Purpose:**

- **Pattern Discovery**: The primary purpose of unsupervised learning is to find hidden patterns or intrinsic structures in the data.
- **Clustering**: It groups similar data points together based on their characteristics (e.g., customer segmentation).
- **Dimensionality Reduction**: It reduces the number of variables (features) in the data while retaining its essential information (e.g., PCA).

**How it works:**

- **Training with Unlabeled Data**: In unsupervised learning, the model is given data without explicit instructions on what to do with it. There are no labels, and the model tries to learn the patterns and the structure from the data itself.
- **No Feedback Loop**: Since there are no labels, there is no direct way to measure the model’s accuracy during training. The goal is to identify inherent patterns or groupings in the data.

# Geometrical ML

Geometrical Machine Learning (Geometric Deep Learning or GML) is a subfield of machine learning that focuses on learning from data with complex, non-Euclidean structures by leveraging concepts from geometry and topology. It extends traditional machine learning approaches to handle data types that are not naturally represented in the usual vector space, such as graphs, manifolds, and other geometric structures.

## Key Concepts in Geometrical Machine Learning

1. **Non-Euclidean Data:**
   - Traditional machine learning techniques often assume that data resides in a Euclidean space (i.e., flat, vector spaces). However, many real-world data types, like social networks, molecules, or 3D shapes, naturally live in non-Euclidean spaces.
   - Examples include graphs (which represent relationships and connections) and manifolds (which represent surfaces or spaces that locally resemble Euclidean space but may have a more complex global structure).

2. **Graphs and Networks:**
   - One of the most common applications of Geometrical ML is on graph-structured data.
   - Techniques like Graph Neural Networks (GNNs) have been developed to generalize neural networks to graph data, allowing for tasks like node classification, link prediction, and graph classification.

3. **Manifold Learning:**
   - Manifolds are geometric spaces that can be curved or have more complex structures than flat Euclidean spaces. Many high-dimensional datasets can be thought of as lying on a lower-dimensional manifold.
   - Techniques in manifold learning aim to understand the underlying manifold structure of the data, often for dimensionality reduction or data visualization.

4. **Symmetries and Invariances:**
   - Geometric ML often exploits symmetries and invariances in data, which are properties that remain unchanged under certain transformations (e.g., rotation, translation).
   - This is particularly useful in fields like computer vision (e.g., recognizing an object regardless of its orientation) and molecular biology (e.g., analyzing molecules in different conformations).

5. **Applications of Geometrical ML:**
   - **Social Networks:** Analyzing relationships, influence, and communities within social graphs.
   - **Molecular Biology:** Predicting molecular properties, drug discovery, and understanding protein structures.
   - **Computer Vision:** Analyzing 3D shapes and point clouds, understanding images in terms of their underlying geometry.
   - **Physics:** Modeling physical systems with inherent geometric properties, like the orbits of celestial bodies.

## Techniques and Algorithms in Geometrical ML

- **Graph Neural Networks (GNNs):** These are designed to operate on graph-structured data by aggregating and propagating information across nodes based on their connectivity.
- **Convolutional Neural Networks on Graphs:** Extensions of CNNs that apply convolution operations over graph-structured data instead of regular grids like images.
- **Manifold Learning Algorithms:** Methods like t-SNE, UMAP, and Isomap, which help uncover the low-dimensional structure of data lying on high-dimensional manifolds.
- **Geodesic Distances and Metric Learning:** Techniques that measure distances on curved surfaces or manifolds, which are more complex than simple Euclidean distances.

# Dimentionality curse

The "curse of dimensionality" in machine learning refers to the various problems and challenges that arise when working with data in high-dimensional spaces. As the number of features (dimensions) in a dataset increases, several issues can occur that make it difficult to train effective models.

## Key Aspects of the Curse of Dimensionality

1. **Sparsity of Data:**
   - In high-dimensional spaces, data points become increasingly sparse. Imagine a dataset with many features (dimensions); as the number of dimensions increases, the volume of the space grows exponentially. This means that the data points are spread out over a much larger space, making it harder for models to find patterns or relationships between the features.

2. **Increased Computational Complexity:**
   - High-dimensional data requires more computational resources. As the number of dimensions increases, the complexity of algorithms also increases, often exponentially.

3. **Overfitting:**
   - With more dimensions, models are more likely to overfit the training data. In high dimensions, models have more freedom to create complex decision boundaries that fit the training data perfectly, but they fail to generalize to new, unseen data.

4. **Difficulty in Visualization and Interpretation**

5. **Distance Measures Lose Meaning:**
   - In high-dimensional spaces, traditional distance measures like Euclidean distance become less meaningful. In low-dimensional spaces, the difference between the nearest and farthest data points is significant. However, in high dimensions, all points tend to become equidistant from each other, making it difficult for distance-based algorithms (like k-nearest neighbors) to function effectively.
   - This phenomenon can lead to the problem where all points appear equally close or far, thus diminishing the effectiveness of algorithms that rely on proximity.

## Strategies to Mitigate the Curse of Dimensionality

1. **Dimensionality Reduction:**

2. **Feature Selection:**

3. **Regularization:**

4. **Using Simpler Models:**

5. **Use Lower powers of Minkowski metrics**

# Manifold

A **manifold** is a mathematical concept used to generalize ideas of shapes and surfaces into higher dimensions. In the context of geometry, topology, and machine learning, manifolds are used to describe spaces that locally resemble Euclidean space but may have a more complex global structure.

## Key Concepts of Manifolds

1. **Locally Euclidean:**
   - A manifold is a space that, around any small neighborhood of a point, looks like regular Euclidean space (i.e., a flat space like a plane in 2D, a line in 1D, or ordinary 3D space). This means that for any point on a manifold, you can zoom in close enough that the space appears to be flat and can be described using standard coordinates.
   - For example, the surface of the Earth (a sphere) can be locally approximated as flat (as on a map), even though globally, it's curved.

2. **Global Structure:**
   - While each small neighborhood of a point on a manifold is Euclidean, the manifold as a whole can have a more complex, curved structure. The global shape of a manifold can be anything from a simple curve or surface to a much more complex structure in higher dimensions.
   - A classic example is the surface of a torus (a doughnut-shaped surface), which is locally flat (like a piece of paper) but globally has a curved and connected structure.

3. **Dimensionality:**
   - The dimension of a manifold is the number of coordinates needed to describe it locally. For example, a line is a 1-dimensional manifold, a plane or the surface of a sphere is 2-dimensional, and our 3D space is a 3-dimensional manifold.
   - In machine learning and data analysis, we often deal with manifolds embedded in higher-dimensional spaces. For instance, the surface of a 3D sphere is a 2-dimensional manifold because locally, you only need two coordinates (like latitude and longitude) to describe a point on the sphere.

## Manifolds in Machine Learning

Manifolds play a significant role in machine learning, especially in high-dimensional data analysis, where the data often lies on a lower-dimensional manifold within a higher-dimensional space.

1. **Manifold Hypothesis:**
   - The manifold hypothesis is the assumption that high-dimensional data (like images, text, or sensory data) often lies on a low-dimensional manifold within the higher-dimensional space. This means that although the data might be represented in a high-dimensional space, the intrinsic structure of the data is lower-dimensional.
   - For example, although an image might have thousands of pixels (features), the set of all possible images of a particular object (like handwritten digits) might lie on a manifold of much lower dimensionality.

2. **Dimensionality Reduction:**
   - Many dimensionality reduction techniques aim to discover and exploit the underlying manifold structure of the data.

3. **Non-Linear Data Structures:**
   - Manifolds allow machine learning models to capture complex, non-linear relationships in data. For instance, when modeling data that lies on a curved surface (a manifold), linear models might struggle, but by understanding the manifold structure, more sophisticated models can better capture these relationships.
   - Manifold learning techniques like Isomap and Locally Linear Embedding (LLE) are specifically designed to map high-dimensional data onto lower-dimensional manifolds while preserving the essential geometric and topological properties of the data.

# Latent (embedding) space

**Latent space** is a concept in machine learning and data analysis, particularly in the context of neural networks, generative models, and dimensionality reduction techniques. It refers to a lower-dimensional space where complex data can be represented in a more compact, abstract, and often meaningful way. The points in this space are called "latent variables" or "latent representations."

## Key Concepts of Latent Space

1. **Dimensionality Reduction:**
   - Latent space is often a lower-dimensional space compared to the original data space.
   - Techniques like Principal Component Analysis (PCA), autoencoders, and variational autoencoders (VAEs) reduce the dimensionality of data by mapping it to a latent space where the main factors of variation are captured.

2. **Abstract Representation:**
   - Latent space captures the underlying structure or the essential features of the data in a more abstract form. In this space, similar data points are usually close together, and different aspects of the data can be disentangled.
   - For example, in a latent space representing images of faces, one dimension might correspond to the person's age, another to their gender, and another to the orientation of their face.

3. **Learning and Mapping:**
   - Neural networks, especially autoencoders and generative models, learn to map data from the original high-dimensional space to a latent space during training. The encoder part of an autoencoder, for example, compresses the input data into a latent representation, and the decoder reconstructs the original data from this latent representation.

4. **Generative Models:**
   - In generative models like Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs), latent space plays a crucial role. These models learn to generate new data by sampling points from the latent space and then transforming them back into the data space.
   - For instance, a VAE trained on images might learn a latent space where each point corresponds to a different image. By sampling from this latent space, the VAE can generate new, similar images.

5. **Interpretability:**
   - Latent spaces can sometimes be interpretable, meaning the dimensions of the latent space correspond to meaningful factors of variation in the data. For example, in a well-learned latent space of human faces, one axis might correspond to facial expressions, another to lighting conditions, and so on.
   - However, in many cases, the dimensions of latent space are abstract and don't directly correspond to any obvious feature of the data.

## Applications of Latent Space

1. **Dimensionality Reduction:**
   - Latent spaces are used in dimensionality reduction techniques like PCA and t-SNE, where high-dimensional data is projected into a lower-dimensional latent space that captures the most important information.

2. **Feature Learning:**
   - Latent spaces are used for learning compact and informative representations of data. For instance, in deep learning, the hidden layers of a neural network can be viewed as encoding the input data into different latent spaces, with each layer capturing more abstract features.

3. **Generative Modeling:**
   - In VAEs and GANs, the latent space is crucial for generating new data points. For example, in a GAN, the generator learns to map random noise from the latent space into realistic data (e.g., images), while the discriminator tries to distinguish between real and generated data.

4. **Data Interpolation and Synthesis:**
   - Latent spaces allow for smooth interpolation between different data points. For example, in the latent space of a VAE trained on faces, interpolating between two points might produce a sequence of images that morph smoothly from one face to another.

5. **Anomaly Detection:**
   - Latent spaces can be used for anomaly detection. By mapping data into latent space, models can identify points that deviate significantly from the normal data distribution, which might indicate anomalies or outliers.

## Examples of Latent Space in Practice

1. **Autoencoders:**
   - Autoencoders consist of an encoder that compresses input data into a latent space and a decoder that reconstructs the original data from this latent space. The latent space captures the essential features needed to reconstruct the data.

2. **Variational Autoencoders (VAEs):**
   - VAEs learn a probabilistic mapping from the data space to a latent space. In the latent space, similar data points are grouped together, and new data can be generated by sampling from this space.

3. **t-SNE and UMAP:**
   - These techniques are used for visualizing high-dimensional data by projecting it into a 2D or 3D latent space that preserves the structure of the data. This allows for the exploration of patterns and clusters in the data.

4. **Generative Adversarial Networks (GANs):**
   - GANs generate new data by learning a mapping from a simple latent space (usually random noise) to the data space. The generator creates data samples from latent space, and the discriminator evaluates their realism.

# Word2Vec

Word2vec is a method to embed words from text corpus into linear space.

**Word2Vec** is a popular technique for natural language processing (NLP) that transforms words into continuous vector representations, typically in a high-dimensional space. These vectors capture semantic information about the words, meaning that words with similar meanings are mapped to nearby points in the vector space.

## How Word2Vec Works

Word2Vec consists of two main model architectures:

1. **Continuous Bag of Words (CBOW)**
2. **Skip-Gram**

Both architectures are based on shallow neural networks and are trained using large corpora of text data. The main idea is to learn a dense vector (embedding) for each word in the vocabulary by predicting either the current word based on its context (CBOW) or the context words based on the current word (Skip-Gram).

### 1. Continuous Bag of Words (CBOW)

- **Objective:** Predict the target word given its surrounding context words.
  
- **Mechanism:**
  - The model takes as input the surrounding words (context) of a target word and tries to predict the target word itself.
  - For example, given the context words "the", "quick", "brown", "fox" around the target word "jumps", the CBOW model tries to predict "jumps".
  - The context words are averaged (or summed) to produce a single vector, which is then fed into the neural network to predict the target word.

- **Use Case:** CBOW is generally faster to train because it takes multiple context words at once, but it can be less effective when the context size is small.

### 2. Skip-Gram

- **Objective:** Predict the surrounding context words given a target word.

- **Mechanism:**
  - The model takes a single word as input and tries to predict the words in its context.
  - For example, given the target word "jumps", the Skip-Gram model tries to predict "the", "quick", "brown", and "fox" as context words.
  - The neural network tries to maximize the probability of context words appearing near the target word in the corpus.
  - The idea is to position the embeddings in such a way that words appearing in similar contexts are close to each other in the vector space.

- **Use Case:** Skip-Gram is more effective for smaller datasets and can better handle rare words, but it can be slower to train because each input word is paired with multiple context words, resulting in more computations.

## Word2Vec Vector Representations

- **Dense Vectors:** The vectors produced by Word2Vec are dense, meaning they contain real-valued numbers, typically in hundreds of dimensions (e.g., 100, 200, or 300 dimensions).

- **Semantic Relationships:** The key feature of Word2Vec vectors is that they capture semantic relationships between words. For example, the vectors for "king" and "queen" are similar, as are the vectors for "Paris" and "France". Moreover, these relationships can often be captured through vector arithmetic:
  - For example, the result of the vector operation **Vec("King") - Vec("Man") + Vec("Woman")** is close to **Vec("Queen")**.

- **Embedding Space:** The vector space is structured in such a way that words with similar meanings or that occur in similar contexts are located near each other. This structure enables various downstream NLP tasks like clustering, classification, and semantic similarity computations.

## Training Word2Vec

- **Corpus:** Word2Vec is typically trained on large text corpora, such as Wikipedia, news articles, or specific domain-related texts.

- **Negative Sampling:** To efficiently train the model, especially for Skip-Gram, a technique called **negative sampling** is often used. Instead of updating all the weights for all the words in the vocabulary (which can be computationally expensive), negative sampling updates the weights for a small subset of "negative" samples that are not the context words.

- **Hierarchical Softmax:** Another optimization technique is **hierarchical softmax**, which speeds up the prediction process by representing the probability distribution over the vocabulary using a binary tree structure.

## Limitations of Word2Vec

- **Static Embeddings:** Word2Vec produces a single embedding for each word, meaning that the representation of a word is the same regardless of its context. This can be limiting for words with multiple meanings (polysemy).

- **Context-Agnostic:** Since it doesn't take into account the specific context in which a word appears, Word2Vec might not capture nuanced meanings that vary depending on context.

- **Data-Hungry:** Word2Vec requires a large amount of data to produce high-quality embeddings. It might not perform well on smaller datasets.

# IsoMap and LLE

**Isomap** and **Locally Linear Embedding (LLE)** are two widely used non-linear dimensionality reduction techniques in machine learning. Both methods aim to reduce the dimensionality of high-dimensional data while preserving the underlying structure, especially when the data lies on a lower-dimensional manifold within the higher-dimensional space.

## Isomap (Isometric Mapping)

**Isomap** is an extension of classical Multidimensional Scaling (MDS) (the example of MDS is the PCA (makes the linear mapping to lower dimention)) that aims to preserve the geodesic distances between data points rather than the straight-line (Euclidean) distances. This makes Isomap particularly effective for data that lies on a curved manifold.

### How Isomap Works

1. **Construct a Neighborhood Graph:**
   - For each data point, identify its neighbors based on some distance metric (usually Euclidean distance). The most common methods for selecting neighbors are:
     - **k-Nearest Neighbors (k-NN):** Connect each point to its k-nearest neighbors.
     - **ε-neighborhood:** Connect each point to all points within a fixed radius ε.
   - The data points are connected by edges representing the distance between them, forming a graph.

2. **Compute Geodesic Distances:**
   - The geodesic distance between any two points on the manifold is approximated by the shortest path distance in the neighborhood graph. This is typically done using algorithms like Dijkstra's or Floyd-Warshall to find the shortest paths between all pairs of points in the graph.

3. **Apply Multidimensional Scaling (MDS):**
   - After computing the geodesic distance matrix, classical MDS is applied to project the data points into a lower-dimensional space. MDS attempts to find a lower-dimensional embedding that best preserves the pairwise geodesic distances.

4. **Output:**
   - The result is a lower-dimensional representation of the data that maintains the intrinsic geometry of the manifold, as captured by the geodesic distances.

### Advantages of Isomap

- **Preserves Global Structure:** Isomap is effective at preserving the global geometric structure of the data, making it suitable for data lying on non-linear manifolds.
- **Non-Linear Relationships:** It captures non-linear relationships between data points that linear methods like PCA cannot.

### Limitations of Isomap

- **Computational Complexity:** The need to compute shortest paths between all pairs of points can be computationally expensive, especially for large datasets.
- **Neighborhood Sensitivity:** The quality of the embedding depends on the choice of neighborhood size (k or ε). Poor choice can lead to disconnected graphs or loss of important structures.

## Locally Linear Embedding (LLE)

**Locally Linear Embedding (LLE)** is another non-linear dimensionality reduction technique that focuses on preserving local relationships among data points. LLE assumes that each data point and its neighbors lie on or near a locally linear patch of the manifold, and it seeks to map these patches to a lower-dimensional space while preserving their local geometries.

### How LLE Works

1. **Construct a Neighborhood Graph:**
   - Similar to Isomap, LLE begins by identifying the nearest neighbors for each data point. The most common method is to use k-nearest neighbors (k-NN).

2. **Compute Local Weights:**
   - For each data point, LLE computes the weights that best reconstruct the data point as a linear combination of its neighbors. These weights are found by minimizing the reconstruction error, ensuring that the weights sum to one and that the reconstruction error is minimized for each point.
   - This step captures the local linear relationships among the data points.

3. **Compute the Lower-Dimensional Embedding:**
   - The goal is to find a lower-dimensional embedding where the same weights used to reconstruct each point from its neighbors in the high-dimensional space can be applied to reconstruct the point in the lower-dimensional space.
   - This is done by minimizing a cost function that measures the discrepancy between the original weights and those in the lower-dimensional space. This step involves solving a sparse eigenvalue problem.

4. **Output:**
   - The result is a lower-dimensional representation of the data that preserves the local geometric properties, ensuring that data points close to each other in the original space remain close in the reduced space.

### Advantages of LLE

- **Preserves Local Structure:** LLE excels at preserving the local structure of the data, making it well-suited for uncovering complex, non-linear manifolds.
- **No Distance Metric Dependency:** Unlike Isomap, LLE does not rely on the global distance metric, which can be advantageous for certain types of data.

### Limitations of LLE

- **Sensitive to Noise:** LLE can be sensitive to noise in the data, which might distort the local relationships and lead to poor embeddings.
- **Neighborhood Size Sensitivity:** Like Isomap, the choice of neighborhood size (k) is crucial and can impact the quality of the embedding.

## Comparison Between Isomap and LLE

- **Global vs. Local Structure:**
  - **Isomap** focuses on preserving global geodesic distances, making it better for capturing the overall geometry of the data manifold.
  - **LLE** emphasizes preserving local linear relationships, which makes it more effective for uncovering local structures and patterns within the data.

- **Computational Complexity:**
  - **Isomap** requires computing shortest paths for all pairs of points, which can be computationally expensive.
  - **LLE** involves solving an eigenvalue problem, which can be less expensive but still computationally intensive for large datasets.

- **Sensitivity to Parameters:**
  - Both methods are sensitive to the choice of neighborhood size, which can significantly affect the resulting embeddings.

- **Applicability:**
  - **Isomap** is often better suited for data where the global structure is important and should be preserved.
  - **LLE** is ideal for scenarios where the local relationships between data points are of primary interest.

# UMAP, Ivis, SNE, t-SNE

Dimensionality reduction methods based on neural networks and other advanced techniques are powerful tools for reducing the complexity of high-dimensional data while preserving important structures and patterns.

## 1. UMAP (Uniform Manifold Approximation and Projection)

### How UMAP Works

1. **Graph Construction:**
   - UMAP starts by constructing a graph representation of the data. Each data point is connected to its nearest neighbors, forming a weighted graph. The weights reflect the probability that a point belongs to the same local structure as its neighbors.

2. **Optimization:**
   - UMAP then optimizes the layout of the graph in a lower-dimensional space by minimizing the difference between the high-dimensional graph and its low-dimensional counterpart. This is done using stochastic gradient descent, with the objective of preserving the topological structure of the original data.

3. **Embedding:**
   - The final result is a lower-dimensional embedding that captures both local neighborhoods and the global structure of the data, making UMAP useful for visualization and further analysis.

### Advantages of UMAP

- **Preserves Global and Local Structures:** UMAP effectively captures both local relationships and the broader global structure of the data, making it versatile.
- **Speed and Scalability:** UMAP is faster and more scalable than t-SNE, especially for large datasets.
- **Flexibility:** UMAP allows for different distance metrics and has parameters that can be tuned to emphasize global or local structures.

### Limitations of UMAP

- **Parameter Sensitivity:** UMAP's performance can be sensitive to parameters like the number of neighbors and the minimum distance, which may require careful tuning.
- **Non-Deterministic:** Due to its stochastic nature, different runs of UMAP on the same data can produce slightly different results.

## 2. Ivis

**Ivis** is a neural network-based dimensionality reduction method that focuses on scalability and the ability to handle very large datasets. It leverages siamese neural networks to learn an embedding that preserves the distances between points in the original space.

### How Ivis Works

1. **Siamese Neural Networks:**
   - Ivis uses siamese neural networks, which are a type of neural network architecture that learns to distinguish between pairs of input data points. It takes pairs of data points as input and learns to output similar embeddings for similar pairs and dissimilar embeddings for dissimilar pairs.

2. **Triplet Loss:**
   - The training process is guided by a triplet loss function, which ensures that the distance between an anchor point and a positive point (a neighbor) is minimized, while the distance between the anchor and a negative point (a non-neighbor) is maximized.

3. **Embedding:**
   - After training, the network generates a lower-dimensional embedding for the data, preserving the neighborhood structure.

### Advantages of Ivis

- **Scalability:** Ivis is designed to handle very large datasets, making it suitable for big data applications.
- **Flexibility:** It can be applied to different types of data, including text and images, by adjusting the neural network architecture.

### Limitations of Ivis

- **Complexity:** Ivis involves training a neural network, which can be computationally intensive and require careful tuning of hyperparameters.
- **Interpretability:** Like other neural network-based methods, the embeddings produced by Ivis may be less interpretable compared to traditional methods.

## 3. SNE (Stochastic Neighbor Embedding)

### How SNE Works

1. **Probability Distribution in High-Dimensional Space:**
   - For each data point, SNE defines a probability distribution over all other points, where the probability is high if the points are close to each other in the original space and low if they are far apart.

2. **Probability Distribution in Low-Dimensional Space:**
   - SNE then defines a similar probability distribution in the lower-dimensional space. The goal is to ensure that these probabilities in the low-dimensional space are as close as possible to those in the high-dimensional space.

3. **Cost Function:**
   - SNE minimizes a cost function (Kullback-Leibler divergence) that measures the difference between the high-dimensional and low-dimensional probability distributions. The optimization process adjusts the embedding in the lower-dimensional space to reduce this cost.

### Advantages of SNE

- **Preservation of Local Structure:** SNE focuses on preserving the local neighborhood structure, making it effective for revealing clusters or local patterns.

### Limitations of SNE

- **Crowding Problem:** SNE can suffer from the "crowding problem," where points in the low-dimensional space are too close together, making it difficult to preserve all neighborhood relationships.
- **Complexity:** The method is computationally intensive, especially for large datasets, and can be slow to converge.

## 4. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**t-SNE** is an improved version of SNE that addresses some of its limitations, particularly the crowding problem. It has become one of the most popular dimensionality reduction techniques for visualizing high-dimensional data.

### How t-SNE Works

1. **Probability Distributions:**
   - Like SNE, t-SNE starts by converting high-dimensional distances between data points into probabilities. However, t-SNE uses a symmetric version of the probability distribution and a heavy-tailed Student's t-distribution in the low-dimensional space to better manage the crowding problem.

2. **Minimizing Divergence:**
   - t-SNE minimizes the Kullback-Leibler divergence between these probability distributions in the high-dimensional and low-dimensional spaces, focusing on preserving the local structure while allowing for more spread in the embedding.

3. **Optimization:**
   - The optimization is performed using gradient descent, resulting in a low-dimensional representation where similar points remain close together, and dissimilar points are farther apart.

### Advantages of t-SNE

- **Effective Visualization:** t-SNE is highly effective for visualizing high-dimensional data in 2 or 3 dimensions, making it a popular tool for exploring and understanding data.
- **Local Structure:** t-SNE excels at preserving local neighborhood relationships, revealing patterns, and clusters in the data.

### Limitations of t-SNE

- **Scalability:** t-SNE can be computationally expensive, particularly for large datasets, and is not as scalable as UMAP or Ivis.
- **Parameter Sensitivity:** t-SNE requires careful tuning of parameters like perplexity, and different settings can lead to very different embeddings.
- **Global Structure:** t-SNE focuses primarily on local structure, sometimes at the expense of global relationships in the data.

## Summary

- **UMAP** is a fast and scalable method that captures both local and global structures, making it suitable for large datasets and various applications.
- **Ivis** uses neural networks and triplet loss to create embeddings that preserve neighborhood structures, particularly suited for big data scenarios.
- **SNE** focuses on preserving local neighborhood relationships but can suffer from the crowding problem and high computational costs.
- **t-SNE** builds on SNE by using a t-distribution to alleviate crowding issues, making it excellent for visualizing high-dimensional data, although it may struggle with scalability and global structure preservation.

# Cross-entropy and Kullback-Leiber divergence

**Cross-entropy** and **Kullback-Leibler (KL) divergence** are both measures used in information theory and machine learning to quantify the difference between probability distributions. While they are related concepts and often used together, they have distinct definitions and serve different purposes.

## 1. Cross-Entropy

**Cross-entropy** measures the difference between two probability distributions by quantifying the expected number of bits required to encode data from one distribution using the code optimized for another distribution. It is often used as a loss function in classification tasks where we want to measure how well the predicted probability distribution $ q $ (e.g., the output of a neural network) matches the true probability distribution $ p $ (e.g., the true labels).

### Definition

For discrete probability distributions $ p $ and $ q $ over a set of events $ X $:

$$
H(p, q) = -\sum_{x \in X} p(x) \log q(x)
$$

Here:

- $ p(x) $ is the true probability of event $ x $.
- $ q(x) $ is the predicted probability of event $ x $.

**Interpretation:**

- Cross-entropy measures how much information is lost when the true distribution $ p $ is encoded using the distribution $ q $.
- It combines the entropy of the true distribution $ p $ and an additional term that measures how $ q $ diverges from $ p $.

## 2. Kullback-Leibler Divergence

**Kullback-Leibler (KL) divergence** measures the difference between two probability distributions by quantifying how much information is lost when one distribution is used to approximate another. Unlike cross-entropy, KL divergence explicitly measures the "distance" from one distribution to another, though it is not a true metric because it is not symmetric.

### Definition

For discrete probability distributions $ p $ and $ q $ over a set of events $ X $:

$$
D_{KL}(p \parallel q) = \sum_{x \in X} p(x) \log \frac{p(x)}{q(x)}
$$

Here:

- $ p(x) $ is the true probability of event $ x $.
- $ q(x) $ is the approximating probability of event $ x $.

**Interpretation:**

- KL divergence measures the "extra" number of bits required to encode samples from $ p $ using the distribution $ q $ instead of the optimal encoding based on $ p $.
- $ D_{KL}(p \parallel q) $ is always non-negative and equals zero only when $ p = q $.

## Relationship Between Cross-Entropy and KL Divergence

The cross-entropy can be broken down into two parts: the entropy of the true distribution and the KL divergence between the true and predicted distributions.

$$
H(p, q) = H(p) + D_{KL}(p \parallel q)
$$

Where:

- $ H(p) = -\sum_{x \in X} p(x) \log p(x) $ is the entropy of the true distribution $ p $, which measures the inherent uncertainty in $ p $.
- $ D_{KL}(p \parallel q) $ measures the difference between $ p $ and $ q $.

Thus, **cross-entropy** combines the inherent uncertainty (entropy) of the true distribution with the "penalty" for diverging from the true distribution (KL divergence).

## Key Differences

1. **Purpose:**
   - **Cross-Entropy:** Measures the total "cost" of encoding data from distribution $ p $ using the distribution $ q $. It's directly used as a loss function in machine learning models.
   - **KL Divergence:** Measures how much one distribution diverges from another, often used to assess the quality of a probabilistic model or to regularize models.

2. **Additivity:**
   - **Cross-Entropy:** Consists of the true distribution's entropy plus the KL divergence between the distributions.
   - **KL Divergence:** Is a component of cross-entropy and directly measures the difference between two distributions.

# k-means and k-means++

**K-means** and **K-means++** are both clustering algorithms used to partition a dataset into $ k $ distinct clusters, where each data point belongs to the cluster with the nearest mean. The key difference between them lies in how they initialize the cluster centroids.

## K-means

1. **Initialization:** Randomly select $ k $ initial cluster centroids.
2. **Assignment Step:** Assign each data point to the nearest centroid, forming $ k $ clusters.
3. **Update Step:** Recalculate the centroids of the clusters by taking the mean of all points in each cluster.
4. **Repeat:** Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.

**Limitations:**

- The quality of the final clusters depends heavily on the initial placement of the centroids. Poor initialization can lead to suboptimal clusters or slow convergence.

## K-means++

**How K-means++ Works:**

1. **Initialization:**
   - Start by randomly selecting the first centroid from the data points.
   - For each subsequent centroid, select a data point with a probability proportional to its squared distance from the nearest existing centroid. This step ensures that new centroids are chosen to be far from the already selected ones.

2. **Proceed with K-means:**
   - After initializing the centroids, the algorithm proceeds with the standard K-means steps (assignment and update).

**Advantages:**

- **Better Initialization:** K-means++ typically results in better clustering outcomes by reducing the chances of poor initialization.
- **Faster Convergence:** Because the initial centroids are more strategically placed, K-means++ often converges faster than K-means.

# DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is a popular clustering algorithm that is particularly effective for identifying clusters of arbitrary shapes and handling outliers or noise in the data. Unlike K-means, which requires the number of clusters to be specified in advance and assumes spherical clusters, DBSCAN groups together points that are closely packed and marks points in low-density regions as outliers.

## How DBSCAN Works

DBSCAN relies on two key parameters:

- **$\epsilon$ (eps):** The maximum distance between two points for them to be considered neighbors.
- **minPts:** The minimum number of points required to form a dense region (i.e., a cluster).

The algorithm classifies data points into three categories:

1. **Core Points:** A point is a core point if it has at least `minPts` points (including itself) within a distance $\epsilon$. Core points are part of a cluster and can potentially expand the cluster.

2. **Border Points:** A point is a border point if it has fewer than `minPts` points within $\epsilon$ but is within the $\epsilon$ distance of a core point. Border points are on the edge of a cluster but do not contribute to its expansion.

3. **Noise Points:** A point is considered noise (or an outlier) if it is neither a core point nor a border point. Noise points are not assigned to any cluster.

## DBSCAN Algorithm Steps

1. **Select an Unvisited Point:**
   - Start with an arbitrary point in the dataset.

2. **Check if It’s a Core Point:**
   - If the point has at least `minPts` neighbors within a distance of $\epsilon$, it’s a core point and forms the seed of a new cluster.

3. **Expand the Cluster:**
   - All points within $\epsilon$ of the core point are added to the cluster. If these points are also core points, their neighbors are recursively added to the cluster.

4. **Mark Border and Noise Points:**
   - Any points within $\epsilon$ of a core point that are not core points themselves are classified as border points. Points that cannot be reached by any core point are marked as noise.

5. **Repeat:**
   - The process is repeated for all unvisited points until all points have been visited.

## Advantages of DBSCAN

- **Identifies Arbitrary Shaped Clusters:** DBSCAN can find clusters of any shape, as long as they are dense enough, making it more flexible than algorithms like K-means.
- **Handles Noise:** DBSCAN effectively identifies outliers or noise, which are simply the points not belonging to any cluster.
- **No Need to Specify the Number of Clusters:** Unlike K-means, DBSCAN does not require the number of clusters to be predefined, which can be advantageous when the number of clusters is not known in advance.

## Limitations of DBSCAN

- **Parameter Sensitivity:** The results of DBSCAN can be sensitive to the choice of $\epsilon$ and `minPts`. If $\epsilon$ is too small, many points will be classified as noise; if too large, clusters may merge together.
- **Difficulty with Varying Densities:** DBSCAN may struggle with datasets where clusters have varying densities, as a single $\epsilon$ value may not be appropriate for all clusters.
- **High-Dimensional Data:** The effectiveness of DBSCAN can diminish in high-dimensional spaces where the concept of distance becomes less meaningful (a problem known as the "curse of dimensionality").

# Hierarchical clustering

**Hierarchical clustering** is a method of cluster analysis that seeks to build a hierarchy of clusters. It creates a tree-like structure (called a dendrogram) that represents how clusters are merged or split at various levels of similarity or distance.

## Types of Hierarchical Clustering

There are two main types of hierarchical clustering:

1. **Agglomerative Hierarchical Clustering (Bottom-Up):**
   - **Process:** Starts with each data point as its own cluster. The algorithm then iteratively merges the closest pairs of clusters until all data points belong to a single cluster or until a desired level of granularity is reached.
   - **Steps:**
     1. Compute the distance (or similarity) between all pairs of clusters.
     2. Merge the two clusters that are closest to each other.
     3. Update the distance matrix to reflect the distance between the new cluster and the remaining clusters.
     4. Repeat steps 2-3 until all points are in one cluster or a stopping criterion is met.

2. **Divisive Hierarchical Clustering (Top-Down):**
   - **Process:** Starts with all data points in a single cluster and then recursively splits the clusters into smaller clusters until each data point is its own cluster or until a certain number of clusters is achieved.
   - **Steps:**
     1. Begin with a single cluster containing all data points.
     2. Choose the best cluster to split into two smaller clusters.
     3. Repeat the process of splitting until the desired number of clusters is obtained or no further splits are possible.

## Dendrograms

A **dendrogram** is a tree-like diagram that records the sequences of merges or splits in hierarchical clustering. It visually represents the hierarchical relationships between the clusters and helps in understanding the structure of the data.

# Clustering metrics

Clustering metrics are used to evaluate the quality of clustering results by measuring how well the algorithm has grouped similar data points together and separated dissimilar ones. Since clustering is an unsupervised learning task, the evaluation can be challenging, but several metrics are commonly used. These metrics can be categorized into internal, external, and relative criteria.

## 1. **Internal Metrics**

Internal metrics assess the quality of the clustering based on the data itself, without reference to external labels.

- **Silhouette Score:**
  - Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
  - The silhouette score ranges from -1 to +1:
    - A score close to +1 indicates that the data points are well matched to their own cluster and poorly matched to neighboring clusters.
    - A score close to 0 indicates that the data point is on or very close to the decision boundary between two neighboring clusters.
    - A score close to -1 indicates that the data points might have been assigned to the wrong cluster.
  - **Formula:**
    $$
    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
    $$
    Where:
    - $ a(i) $ is the average distance from the data point $ i $ to all other points in the same cluster.
    - $ b(i) $ is the average distance from the data point $ i $ to all points in the nearest different cluster.

- **Davies-Bouldin Index:**
  - Measures the average similarity ratio of each cluster with its most similar cluster, based on the ratio of within-cluster distances to between-cluster distances.
  - The lower the Davies-Bouldin Index, the better the clustering.
  - **Formula:**
    $$
    DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)
    $$
    Where:
    - $ \sigma_i $ is the average distance between each point in the cluster $ i $ and the centroid $ c_i $.
    - $ d(c_i, c_j) $ is the distance between the centroids of clusters $ i $ and $ j $.

- **Dunn Index:**
  - Measures the ratio of the minimum inter-cluster distance to the maximum intra-cluster distance.
  - A higher Dunn Index indicates better clustering.
  - **Formula:**
    $$
    D = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_k d(C_k)}
    $$
    Where:
    - $ d(C_i, C_j) $ is the distance between clusters $ i $ and $ j $.
    - $ d(C_k) $ is the intra-cluster distance for cluster $ k $.

## 2. **External Metrics**

External metrics require ground truth labels and compare the clustering results to the true labels to evaluate the clustering quality.

- **Adjusted Rand Index (ARI):**
  - Measures the similarity between the clusters produced by the algorithm and the ground truth clusters, adjusting for chance.
  - The ARI ranges from -1 to 1, where 1 indicates perfect clustering, 0 indicates random clustering, and negative values indicate worse than random.
  - **Formula:**
    $$
    \text{ARI} = \frac{\text{RI} - \text{Expected RI}}{\max(\text{RI}) - \text{Expected RI}}
    $$
    Where RI is the Rand Index, a measure of agreement between two clusterings.

- **Normalized Mutual Information (NMI):**
  - Measures the amount of information shared between the clustering and the ground truth, normalized to account for chance.
  - The NMI ranges from 0 to 1, where 1 indicates perfect agreement.
  - **Formula:**
    $$
    \text{NMI} = \frac{2 \cdot I(X; Y)}{H(X) + H(Y)}
    $$
    Where:
    - $ I(X; Y) $ is the mutual information between the true labels $ X $ and the predicted labels $ Y $.
    - $ H(X) $ and $ H(Y) $ are the entropies of the true and predicted labels, respectively.

- **Fowlkes-Mallows Index (FMI):**
  - Measures the geometric mean of the pairwise precision and recall between the clustering and the ground truth.
  - The FMI ranges from 0 to 1, with 1 indicating perfect clustering.
  - **Formula:**
    $$
    \text{FMI} = \sqrt{\frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}}
    $$
    Where:
    - $ TP $ (True Positives): Pairs of points that are in the same cluster in both the predicted and true clusterings.
    - $ FP $ (False Positives): Pairs of points that are in the same cluster in the predicted clustering but not in the true clustering.
    - $ FN $ (False Negatives): Pairs of points that are in different clusters in the predicted clustering but in the same cluster in the true clustering.

## 3. **Relative Metrics**

Relative metrics compare the results of different clustering algorithms or the same algorithm with different parameter settings, often using internal or external metrics.

- **Elbow Method:**
  - Used to determine the optimal number of clusters by plotting the within-cluster sum of squares (inertia) against the number of clusters. The "elbow" point on the curve suggests the optimal number of clusters.

- **Silhouette Analysis:**
  - Similar to the Silhouette Score but used iteratively for different numbers of clusters to determine the optimal number by selecting the configuration with the highest average silhouette score.

# Expectation-Maximization algorithm

[Слайд 65 в лекции Кантора](clustering_cantor.pdf)

# Kernel Density Estimation

**Kernel Density Estimation (KDE)** is a non-parametric method used in statistics to estimate the probability density function (PDF) of a random variable. Unlike parametric methods that assume the data follows a specific distribution (like a Gaussian distribution), KDE makes no such assumptions, making it a flexible tool for data analysis.

## How Kernel Density Estimation Works

KDE works by placing a smooth "kernel" (a smooth, symmetric function) on each data point and then summing up all these kernels to form a continuous estimate of the density.

### Steps Involved in KDE

1. **Choose a Kernel Function:**
   - The kernel function is usually a symmetric, smooth function that integrates to 1. The most commonly used kernel is the Gaussian (normal) kernel, but other kernels like Epanechnikov, triangular, and uniform can also be used.
   - **Gaussian Kernel Example:**
     $$
     K(u) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}u^2}
     $$
   - In this formula, $ u $ represents the scaled distance between the data point and the point where the density is being estimated.

2. **Choose a Bandwidth (h):**
   - The bandwidth $ h $ controls the width of the kernel and thus the smoothness of the resulting density estimate. A smaller bandwidth results in a more detailed (but potentially noisy) estimate, while a larger bandwidth produces a smoother estimate.
   - Bandwidth selection is crucial, as it determines the balance between bias (smoothness) and variance (detail).

3. **Estimate the Density:**
   - The density at a point $ x $ is estimated by summing the contributions of all the kernels centered at the data points. The KDE formula is:
     $$
     \hat{f}(x) = \frac{1}{n h} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)
     $$
     Where:
     - $ \hat{f}(x) $ is the estimated density at point $ x $.
     - $ n $ is the number of data points.
     - $ x_i $ are the data points.
     - $ h $ is the bandwidth.
     - $ K $ is the kernel function.

# Further readings

- [Good lecture on MDS, Isomap, LLE](https://habr.com/ru/post/321216)
- [Lecture on t-SNE (this one is good too)](https://www.youtube.com/watch?v=4GBgqmq0XAY)
- [Slides about clusterization](https://github.com/vkantor/MIPT_Data_Mining_In_Action_2016/blob/master/lectures/3_Clustering.pdf)
- [Metrics in clusterization](https://github.com/esokolov/ml-course-hse/blob/master/2018-fall/lecture-notes/lecture11-unsupervised.pdf)
- [Slides about ICA](https://www.dropbox.com/s/fhcg7598v048yei/GMML_Bernstein_Lecture3.pdf?dl=0)
- [More clustering methods (in Russian)](https://habr.com/ru/post/321216/)
