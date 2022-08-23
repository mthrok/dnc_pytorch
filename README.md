# Differentiable Neural Computer in PyTorch

This is PyTorch port of DeepMind's [Differentiable Neural Computer (DNC)](https://github.com/deepmind/dnc).

The original code was written with TensorFlow v1, which is not straightforward to set up in modern tech stack, such as Apple Scilicon and Python 3.8+.

This code requires only PyTorch >= 1.10.

The code structure and interfaces are kept almost same.
The original docstring is preserved as-is.

## Usage

### Training

You can run repeat-copy task with `python train.py`.
It will start the training of DNC with repeat-copy task, and generates output like the following.

For the format of the display, please checkout [the docstring of RepeatCopy class](https://github.com/mthrok/dnc_pytorch/blob/96079557968e3a6905ebcf72373f7a70e3ab4c87/dnc/repeat_copy.py#L100-L159).

```
2022-08-23 22:05:58,750: 18499: Avg training loss 2.1131072425842286
2022-08-23 22:05:58,750:
Observations:
+- 1 - - - - - - -+
+- - 1 - - - - - -+
+- 1 - - - - - - -+
+- 1 1 - - - - - -+
+1 - - - - - - - -+
+- - - 1 - - - - -+

Targets:
+- - - - 1 - - - -+
+- - - - - 1 - - -+
+- - - - 1 - - - -+
+- - - - 1 1 - - -+
+- - - - - - 1 - -+

Model Output:
+- - - - 1 - - - -+
+- - - - 1 - - - -+
+- - - - 1 - - - -+
+- - - - 1 1 - - -+
+- - - - - - 1 - -+
```

### Unit tests

Unit tests are also ported. Run `pytest tests`.
