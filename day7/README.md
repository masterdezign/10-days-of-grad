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
Direct Feedback Alignment

...

SGD (zero biases)
1 Training accuracy 11.2  Validation accuracy 11.4
2 Training accuracy 11.2  Validation accuracy 11.4
3 Training accuracy 11.3  Validation accuracy 11.4
4 Training accuracy 24.0  Validation accuracy 24.1
5 Training accuracy 42.7  Validation accuracy 43.5
6 Training accuracy 59.1  Validation accuracy 60.9
7 Training accuracy 70.8  Validation accuracy 72.3
8 Training accuracy 76.9  Validation accuracy 77.9
9 Training accuracy 76.0  Validation accuracy 76.8
10 Training accuracy 79.2  Validation accuracy 80.0
```
