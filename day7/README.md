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
