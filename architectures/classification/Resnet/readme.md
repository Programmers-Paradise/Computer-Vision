
# 📚 **ResNet Implementations for Computer Vision**

Welcome to the ResNet implementation repository! Here, you’ll find various ResNet architectures implemented from scratch in **PyTorch**, including **ResNet-18**, **ResNet-34**, **ResNet-50**, **ResNet-101**, and **ResNet-152**. These are some of the most influential architectures in the field of **deep learning** and **computer vision**.

This repository includes:
- **Implementation of each ResNet model**.
- Explanations of how ResNet works.
- Use cases of ResNet and its variants.
- Links to related research papers and further reading.

---

## 💡 **Overview of ResNet**

**ResNet** (Residual Networks) was introduced in the paper "Deep Residual Learning for Image Recognition" by **Kaiming He et al.** in 2015. ResNet's key innovation is the introduction of **residual learning**, which allows training of very deep networks by avoiding the vanishing gradient problem through **skip connections**.

The key idea is to learn the residual (difference) between input and output, instead of the direct mapping, which makes the training of deeper networks more efficient and effective.

---

## 🚀 **ResNet Architecture**

### **Basic Components of ResNet:**
1. **Residual Blocks**: Each block consists of two or more convolutional layers, where the input is directly added to the output of the convolution, bypassing one or more layers.
2. **Skip Connections**: These connections allow gradients to flow directly across layers, making deep networks trainable.
3. **Bottleneck Blocks (for deeper ResNets)**: These blocks are used in architectures like ResNet-50, ResNet-101, and ResNet-152, where the input is compressed and expanded to reduce computation and parameters.

### **ResNet Variants**:
- **ResNet-18**: Smallest ResNet variant with 18 layers. Ideal for quick experiments and small datasets.
- **ResNet-34**: Slightly larger, with more layers for better performance on standard datasets.
- **ResNet-50**: Uses **Bottleneck blocks**. Suitable for more complex problems.
- **ResNet-101**: Deeper ResNet with 101 layers. Best for highly complex problems.
- **ResNet-152**: The deepest ResNet, used for extremely complex problems and large datasets.

---

## 🏆 **Use Cases of ResNet**

ResNet is widely used across many **computer vision** tasks, including but not limited to:
- **Image Classification**: Classifying objects in images, typically using datasets like **ImageNet**.
- **Object Detection**: Using pre-trained ResNet networks as backbone feature extractors in networks like **Faster R-CNN** and **YOLO**.
- **Image Segmentation**: Using ResNet with techniques like **U-Net** for tasks such as medical image analysis.
- **Facial Recognition**: ResNet has been applied in facial recognition systems due to its robustness in identifying features.
- **Transfer Learning**: ResNet models are often used as pre-trained models, fine-tuned for specific tasks such as image classification or object detection.

---

## 🔬 **Sources for Study**

### **Research Papers**:
- **[Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385)**: The foundational paper introducing ResNet and the idea of residual learning.
- **[Identity Mappings in Deep Residual Networks (ResNet-110)](https://arxiv.org/abs/1603.05027)**: Explores identity mappings in ResNet architectures for even deeper networks.
- **[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)**: A paper that gives practical tips on improving ResNet models for image classification tasks.

### **Books and Tutorials**:
- **[Deep Learning with Python by François Chollet](https://www.manning.com/books/deep-learning-with-python)**: A great resource for learning deep learning concepts and their applications using Keras and TensorFlow.
- **[Stanford CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)**: A comprehensive and in-depth course on computer vision and neural networks by Stanford.
- **[PyTorch Documentation](https://pytorch.org/docs/stable/)**: Official documentation for PyTorch. Essential for understanding and implementing neural networks.

---




## 🔗 **References**

1. **[Original ResNet Paper: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)**
2. **[PyTorch Official Documentation](https://pytorch.org/docs/stable/)**
3. **[Stanford CS231n Course: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)**

---

## 🌟 **Acknowledgments**

Special thanks to **Kaiming He et al.** for their groundbreaking work in developing ResNet. This repository is built upon their work to further research and explore deep learning techniques in computer vision.

