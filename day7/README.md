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

Direct feedback alignment
1 Training accuracy 55.7  Validation accuracy 56.0
2 Training accuracy 73.2  Validation accuracy 75.0
3 Training accuracy 83.1  Validation accuracy 82.8
4 Training accuracy 81.5  Validation accuracy 82.0
5 Training accuracy 86.7  Validation accuracy 87.2
6 Training accuracy 88.4  Validation accuracy 88.7
7 Training accuracy 87.7  Validation accuracy 88.1
8 Training accuracy 86.0  Validation accuracy 86.4
9 Training accuracy 87.8  Validation accuracy 88.4
10 Training accuracy 89.9  Validation accuracy 90.2

(Then, drops back to 0.1)

```
