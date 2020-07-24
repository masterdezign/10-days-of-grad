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
1 Training accuracy 32.7  Validation accuracy 33.0
2 Training accuracy 65.4  Validation accuracy 65.5
3 Training accuracy 81.0  Validation accuracy 81.6


```
