# Activating discriminative semantic for few-shot classification

## General image classification

### Environment
This code requires Pytorch 1.10.0 and torchvision 0.11.0 or higher with cuda support.
***
### Result
mini-ImageNet

| Model    | 5-way 1-shot | 5-way 5-shot |
|----------|--------------|--------------|
| ProtoNet | 62.39±0.21   | 80.53±0.14   |
| Poodle*  | 74.21        | 83.71        |
| Our      | 65.39±0.61   | 81.28±0.43   |
| Our*     | 76.82±0.41   | 82.93±0.38   |

tiered-ImageNet

| Model    | 5-way 1-shot | 5-way 5-shot |
|----------|--------------|--------------|
| ProtoNet | 68.23±0.23   | 84.03±0.16   |
| Poodle*  | 78.72        | 86.57        |
| Our      | 70.03±0.72   | 84.53±0.49   |
| Our*     | 84.41±0.40   | 86.15±0.36   |

***
### Dataset
Download mini-ImageNet, tiered-ImageNet and CUB dataset and put them into ```./``` .

- [mini-ImageNet]()
- [tiered-ImageNet]()
- [CUB]()

***
### Run
For example, to train the 1-shot/5-shot 5-way model with ProtoNet backbone on MiniImageNet:
```python ./Mini_imageNet/pre_train_improve_k.py```

to train the 1-shot/5-shot 5-way model with ProtoNet backbone on tieredImageNet:
```python ./Tiered_imagenet/pre_train_improve_k.py```


