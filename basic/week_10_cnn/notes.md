# Further readings

- [en] Convolutional Neural Networks: Architectures, Convolution / Pooling
  Layers: http://cs231n.github.io/convolutional-networks/
- [en] Understanding and Visualizing Convolutional Neural Networks:
  http://cs231n.github.io/understanding-cnn/
- [en] CS231n notes on data preparation:
  http://cs231n.github.io/neural-networks-2/

# 1x1 convolution

A **1x1 convolution layer** is a special type of convolution operation that uses filters of size $1 \times 1$. Though it may seem trivial, it serves several important purposes in neural networks, particularly in **deep learning** architectures like convolutional neural networks (CNNs).

Here are the key purposes of using a 1x1 convolution layer:

### 1. **Dimensionality Reduction (Channel Reduction)**
   - **Purpose**: A 1x1 convolution can be used to reduce the number of channels (or depth) of the feature maps without affecting the spatial dimensions (width and height).
   - **How**: By applying a $1 \times 1$ convolution filter, you can project the input feature map (which may have a large number of channels) to a smaller number of channels. For example, if you have a feature map with 256 channels, you can use $1 \times 1$ convolutions with 64 filters to reduce it to 64 channels.
   - **Why**: This is especially useful in **bottleneck layers** (e.g., in ResNet architectures) to reduce the computational cost of convolutions by lowering the number of channels before applying more expensive spatial convolutions (like $3 \times 3$ or $5 \times 5$ convolutions).

### 2. **Dimensionality Expansion (Channel Expansion)**
   - **Purpose**: Similarly, 1x1 convolutions can also be used to **increase** the number of feature map channels.
   - **How**: If you apply 1x1 convolution with more filters than the input channels, you can increase the depth of the feature map.
   - **Why**: This can be helpful for enriching the representational capacity of the network without altering the spatial resolution of the feature maps.

### 3. **Non-linear Merging of Information Across Channels**
   - **Purpose**: A $1 \times 1$ convolution enables the network to **mix information across different channels** without affecting the spatial dimensions.
   - **How**: While traditional spatial convolutions (e.g., $3 \times 3$, $5 \times 5$) capture spatial patterns, the $1 \times 1$ convolution works on the **channel dimension**, allowing the network to recombine features and learn new patterns across different channels.
   - **Why**: This can be seen as a form of feature re-weighting or cross-channel interaction. The $1 \times 1$ convolution can apply non-linear activation (like ReLU) after the convolution, allowing the model to create more complex combinations of features learned in previous layers.

### 4. **Increasing Model Depth Without Increasing Complexity**
   - **Purpose**: A 1x1 convolution allows for deeper networks by introducing more layers while keeping the number of parameters relatively low.
   - **How**: A 1x1 convolution followed by a non-linearity (such as ReLU) acts like a fully connected layer applied to each pixel or position in the feature map but with fewer parameters.
   - **Why**: This is particularly useful in architectures like **Inception Networks**, where 1x1 convolutions allow for more depth and richness in the model’s representation without significantly increasing the number of parameters or computational complexity.

### 5. **Improving Computational Efficiency**
   - **Purpose**: Using $1 \times 1$ convolutions can reduce the computational cost of operations in CNNs by compressing channels before more expensive convolutions.
   - **How**: Instead of applying large convolutions directly (e.g., $3 \times 3$ or $5 \times 5$ convolutions with many input channels), a $1 \times 1$ convolution can first be applied to reduce the number of channels, followed by the larger spatial convolution. This technique is commonly used in bottleneck layers to improve efficiency.
   - **Why**: It helps strike a balance between model complexity and computational cost by first reducing the dimensionality with 1x1 convolutions, then applying more complex operations afterward.

### 6. **Cross-Channel Interaction for Global Information**
   - **Purpose**: $1 \times 1$ convolutions allow each pixel's value in a feature map to be influenced by all channels of the input feature map at that location, enabling **cross-channel interaction** without altering spatial relationships.
   - **How**: Each $1 \times 1$ filter looks at every input channel, performs a weighted sum, and produces a single output. This essentially applies a fully connected layer at each spatial location of the input.
   - **Why**: It lets the model aggregate information across different channels to learn more diverse and complex feature representations at each spatial location.

### 7. **Network in Network (NIN)**
   - **Purpose**: $1 \times 1$ convolutions play a key role in architectures like **Network in Network (NIN)**, where they are used to replace fully connected layers by making the network more compact and reducing the number of parameters.
   - **How**: The NIN architecture replaces traditional large fully connected layers with small convolutional layers, typically 1x1, to perform a similar role but in a spatially-aware manner.
   - **Why**: This helps reduce overfitting, decreases the number of parameters, and enables a deeper, more efficient network.

### 8. **Attention Mechanisms**
   - **Purpose**: 1x1 convolutions are also used in attention mechanisms within CNN architectures.
   - **How**: In some cases, 1x1 convolutions are applied after global average pooling or other operations to assign weights (or attention) to different channels or spatial locations.
   - **Why**: This allows the model to focus on the most important features dynamically during training and inference.

### Example Architectures that Use 1x1 Convolutions:
   - **Inception Networks (GoogleNet)**: Inception modules use $1 \times 1$ convolutions to reduce the number of channels before applying more computationally expensive convolutions like $3 \times 3$ or $5 \times 5$.
   - **ResNet (Residual Networks)**: ResNet uses $1 \times 1$ convolutions in its bottleneck layers to compress and then expand channels, making the network more efficient.
   - **SqueezeNet**: SqueezeNet heavily uses $1 \times 1$ convolutions to reduce the number of parameters while maintaining competitive performance on tasks like image classification.

### Summary:
In summary, 1x1 convolutions are used in neural networks for:
- **Reducing or increasing the depth of feature maps** (channel compression or expansion).
- **Mixing information across channels** for better feature learning.
- **Improving computational efficiency** by reducing parameters and operations.
- **Deepening networks** with more non-linear transformations without altering spatial dimensions.

The use of $1 \times 1$ convolutions enhances both the flexibility and efficiency of modern CNN architectures, making them a crucial component of many state-of-the-art models.

# Fine-tuning vs Feature extraction

In **finetuning**, we start with a
pretrained model and update *all* of the model’s parameters for our new
task, in essence retraining the whole model. In **feature extraction**,
we start with a pretrained model and only update the final layer weights
from which we derive predictions. It is called feature extraction
because we use the pretrained CNN as a fixed feature-extractor, and only
change the output layer.
