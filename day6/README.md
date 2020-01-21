# Binarized Neural Networks

The project demonstrates binarized neural network training.
Need some documentation? Here is
[the tutorial](http://penkovsky.com/neural-networks/day6).

## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Build and run MNIST benchmark on all available cores:

     ```
     $ ./run.sh
     ```

```
Training a binarized network
1 Training accuracy 89.5  Validation accuracy 89.3
2 Training accuracy 91.9  Validation accuracy 91.8
3 Training accuracy 93.1  Validation accuracy 92.1
4 Training accuracy 93.5  Validation accuracy 92.7
...
```
