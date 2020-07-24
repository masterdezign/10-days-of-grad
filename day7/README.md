# Direct Feedback Alignment

![Handwritten digit](http://penkovsky.com/img/posts/mnist/mnist-five.png)

This project's aim is to demonstrate direct feedback alignment using handwritten
digits recognition task as an example. Relevant documentation is in
[the tutorial about neural networks](http://penkovsky.com/neural-networks/day7/).

## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Build and run the MNIST benchmark on all available cores:

     ```
     $ ./run.sh
     ```

```
Direct feedback alignment

1 Training accuracy 53.1  Validation accuracy 53.2
2 Training accuracy 83.4  Validation accuracy 83.4
3 Training accuracy 87.6  Validation accuracy 87.6
4 Training accuracy 86.5  Validation accuracy 86.2
5 Training accuracy 89.0  Validation accuracy 89.5
6 Training accuracy 88.8  Validation accuracy 89.7
7 Training accuracy 90.1  Validation accuracy 90.3
8 Training accuracy 90.3  Validation accuracy 90.5
9 Training accuracy 90.7  Validation accuracy 90.6
10 Training accuracy 89.3  Validation accuracy 89.4
11 Training accuracy 89.3  Validation accuracy 89.0
12 Training accuracy 89.4  Validation accuracy 89.5
13 Training accuracy 90.6  Validation accuracy 90.8
14 Training accuracy 89.5  Validation accuracy 90.1
15 Training accuracy 90.3  Validation accuracy 90.5

```
