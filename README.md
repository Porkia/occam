# occam
This is a code reproducing paper :

Zheng, Baolin, et al. “Black-Box Adversarial Attacks on Commercial Speech Platforms with Minimal Information.” Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security, 2021, pp. 86–107. arXiv.org, https://doi.org/10.1145/3460120.3485383.

The target test model is a wac2vec 2.0 model (https://huggingface.co/yongjian/wav2vec2-large-a).

If you want to run this, first install pytorch : https://pytorch.org/
and transformers : https://huggingface.co/docs/transformers/installation

e.g., installation by conda (or just pip)

```
conda create -n wav2vec2 python=3.8
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install -c conda-forge transformers
```

After installation, just run the attack file.
```
python occam.py
```

Change the parameters and filename in 'occam.py' to create adversarial examples as you like.
If you want attack other models or apis, customize the fitness function and pass this to attack function as a parameter.
```python
final_adv = occam.attack( x, x_init_adv, fitness_func )
```
The fitness function should take original input and adversarial input as parameters, and return a fitness value.
