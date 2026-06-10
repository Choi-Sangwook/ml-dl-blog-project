# Deep Learning Post Checklist

Apply this checklist only when the source includes neural networks, tensors, framework setup, GPU use, datasets, or training code.

## 1. Framework And Environment

- Identify the framework for every complete example: PyTorch, TensorFlow/Keras, or another framework.
- Do not mix PyTorch model code with Keras preprocessing or training APIs without an explicit comparison section.
- Treat installation commands as time-sensitive.
- Verify current installation instructions, supported Python versions, accelerator requirements, and platform limitations using official framework documentation.
- State the relevant framework and version when behavior is version-dependent.
- Separate CPU installation from NVIDIA GPU installation.
- Do not imply that CUDA is available on every machine.
- Provide a CPU fallback and guard GPU access.

PyTorch device handling should follow this pattern:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

model = model.to(device)
X = X.to(device)
```

Do not call `torch.cuda.get_device_name(0)` or move tensors to `"cuda"` before checking availability.

Clarify these roles when they appear:

- CUDA: NVIDIA's parallel-computing platform and runtime
- cuDNN: NVIDIA's optimized deep-neural-network library
- framework: PyTorch or TensorFlow/Keras

Avoid presenting them as interchangeable software packages.

## 2. Screenshot And Notebook Recovery

Deep-learning lecture PDFs frequently store code as screenshots.

- Flag pages with fewer than roughly 80 extracted characters when they contain images.
- Inspect consecutive low-text pages as one code-example block.
- Recover imports, data loading, preprocessing, model definition, compile or optimizer setup, training, evaluation, and prediction.
- Check indentation and line wrapping after OCR.
- Do not publish OCR-derived code until syntax and API names are verified.
- If only part of a notebook is visible, label the reconstruction as an added runnable example rather than copied PDF code.

## 3. Tensor Shape And Data Type

For every major code example, document the important shapes.

Examples:

```text
tabular batch: (batch_size, num_features)
MNIST in Keras: (batch_size, 28, 28, 1)
MNIST in PyTorch: (batch_size, 1, 28, 28)
flattened MNIST: (batch_size, 784)
multiclass logits: (batch_size, num_classes)
binary logits: (batch_size,) or (batch_size, 1)
```

Check:

- batch dimension is preserved;
- matrix multiplication dimensions are compatible;
- flattening does not accidentally merge the batch dimension;
- Keras image layout is usually NHWC and PyTorch is usually NCHW;
- input tensors use a floating type expected by the model;
- PyTorch `CrossEntropyLoss` class targets are integer `torch.long`;
- binary targets match the output shape and floating type required by the loss;
- model parameters and input tensors are on the same device.

Do not describe a tensor as merely an array without explaining shape when shape drives the operation.

## 4. Output Layer, Label, And Loss Compatibility

Use compatible combinations.

| Task | Typical model output | Label format | PyTorch loss | Keras loss |
|---|---|---|---|---|
| Regression | raw value(s) | float | `MSELoss`, `L1Loss` | `mse`, `mae` |
| Binary classification | one raw logit | 0/1 float | `BCEWithLogitsLoss` | `BinaryCrossentropy(from_logits=True)` |
| Multiclass, integer labels | class logits | class index | `CrossEntropyLoss` | `SparseCategoricalCrossentropy(from_logits=True)` |
| Multiclass, one-hot labels | class logits or probabilities | one-hot vector | suitable manual setup | `CategoricalCrossentropy` |

Important caveats:

- `BCEWithLogitsLoss` already combines sigmoid and binary cross-entropy. Do not put sigmoid in the model before this loss.
- `CrossEntropyLoss` expects raw logits and internally applies the needed log-softmax calculation. Do not apply softmax before the loss.
- One-hot encoding is not universally required for multiclass classification. It depends on the selected loss and framework.
- Apply sigmoid or softmax for human-readable inference probabilities only when the training loss expects logits.
- Explain binary classification separately from multilabel classification. Multilabel outputs usually use one sigmoid per label, not softmax.

## 5. Model Definition And Training Loop

For PyTorch:

- subclass `nn.Module` correctly and call `super().__init__()`;
- check every `nn.Linear(in_features, out_features)` connection;
- call `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` once per update;
- use `model.train()` for training;
- use `model.eval()` and `torch.no_grad()` for validation and prediction;
- avoid unbounded `while True` searches in tutorial code;
- report parameter count or model summary when the architecture is central.

For Keras:

- show the expected `input_shape`;
- distinguish model construction, `compile`, `fit`, `evaluate`, and `predict`;
- explain `validation_data` or `validation_split`;
- use callbacks such as early stopping or model checkpointing only when they support the lesson;
- restore or evaluate the best validation checkpoint when claiming final performance.

For both:

- distinguish epoch, batch, and optimization step;
- avoid saying a larger batch is always faster or better;
- track training and validation metrics separately;
- evaluate the test set only after model selection is complete;
- explain underfitting and overfitting from both training and validation curves.

## 6. Reproducibility

When reproducibility matters, seed all frameworks actually used:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

State that GPU kernels and parallel execution can still cause small nondeterministic differences. Do not guarantee bit-for-bit equality unless deterministic settings were explicitly configured and tested.

## 7. Dataset And Split Integrity

- Verify dataset source, license, availability, shape, class count, and split sizes.
- Check whether a legacy dataset is deprecated or has ethical concerns.
- Do not introduce Boston Housing as an unqualified default dataset; mention its legacy status and consider a maintained alternative.
- For medical or sensitive datasets, avoid implying clinical validity from a classroom model.
- Explain sensitive attributes and proxy variables when relevant.
- Keep train, validation, and test roles distinct.
- Never tune hyperparameters on the test set.
- Use stratification for classification when appropriate.
- Use group-aware or chronological splitting when samples are related or time ordered.
- Fit normalization statistics and vocabulary only on training data.
- Apply random augmentation only to training data.
- Shuffle training data; do not shuffle merely to make evaluation look random.
- Check duplicates or near-duplicates across splits for image and text datasets.

## 8. Image Dataset Handling

For MNIST, CIFAR-10, and similar data:

- state image height, width, channel count, and value range;
- visualize several samples with labels before training;
- normalize pixel values and explain the transformation;
- keep channel order consistent with the framework;
- use MLP flattening only as an introductory baseline;
- explain that flattening removes spatial structure and CNNs usually suit images better;
- do not claim that CIFAR-100 contains fine-grained dog breeds unless the actual class taxonomy supports that statement;
- distinguish class probability examples from actual model output.

If augmenting images, explain which transformations preserve the label. Avoid transformations that can change the answer.

## 9. Weights, Biases, And Activations

- Explain a weight as a learned coefficient, not a direct universal measure of feature importance.
- Avoid statements such as "weight 0.8 means the feature is exactly eight times more important" without strong assumptions.
- Explain that bias shifts the pre-activation value or decision boundary.
- Distinguish pre-activation `z` from activation output `a`.
- State that stacked linear layers without nonlinear activation collapse into one linear transformation.
- Explain vanishing and exploding gradients without claiming that ReLU completely removes them.
- Describe dead ReLU as a risk, not an inevitable outcome.
- Avoid calling one activation universally optimal.
- Match activation discussion to hidden layers versus output layers.

## 10. Linear Algebra And Multidimensional Arrays

When the PDF includes matrix theory:

- cover only the operations needed to understand tensors and neural-network computation;
- connect matrix shapes to batches, weights, activations, and logits;
- explain elementwise multiplication separately from matrix multiplication;
- explain dot product, outer product, transpose, identity matrix, and broadcasting when used;
- show at least one NumPy or PyTorch shape example;
- do not imply that matrix multiplication is commutative;
- prefer a numerical solver such as `torch.linalg.solve` or `numpy.linalg.solve` over explicitly computing an inverse when solving linear systems;
- explain that deep-learning code rarely computes a matrix inverse by hand;
- move proofs and extensive inverse-matrix exercises to an optional appendix unless they are the lecture's main objective.

Example:

```python
import torch

X = torch.randn(32, 10)   # 32 samples, 10 features
W = torch.randn(10, 64)   # 10 inputs, 64 neurons
b = torch.randn(64)

Z = X @ W + b             # broadcasting: (32, 64) + (64,)
print(Z.shape)             # torch.Size([32, 64])
```

## 11. Claims And Historical Context

- Keep AI history proportional to the post's learning goal.
- Verify dates, researcher contributions, organization names, and framework ownership before publishing.
- Avoid attributing the invention of multilayer networks or a whole field to one person.
- Separate biological inspiration from literal equivalence; artificial neural networks are simplified mathematical models, not replicas of the brain.
- Treat statements such as "most used", "industry standard", or "best" as time-sensitive and qualify or verify them.

## 12. Minimum Deep-Learning Review Output

For a review report, explicitly include:

1. PDF coverage, including screenshot-only pages
2. Framework and environment issues
3. Tensor shape and data type issues
4. Output-layer, label, and loss compatibility
5. Data split and preprocessing leakage
6. Training and evaluation state
7. Dataset source, ethics, and deprecation
8. Overstated historical or technical claims
9. Paste-ready corrections

