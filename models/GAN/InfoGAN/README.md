## Vanilla Generative Adversarial Networks (GANs)

Training notes can be found at: [www.sciencesundries.com/sundries](http://www.sciencesundries.com/sundries)

### GAN:
Running example
```
python Train.py 7 --Net linear --logdir ./runs
```

### DCGAN:
Running example
```
python Train.py 7 --Net conv --logdir ./runs
```

### Logs
Training logs can be viewed using tensorboard:
```
tensorboard --logdir ./runs
```


## REFERENCES

```
[1] Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
      Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. “Generative
      Adversarial Nets.” In Advances in Neural Information Processing Systems,
      2672–2680.http://papers.nips.cc/paper/5423-generative-adversarial-nets.
[2] Radford, A., Metz, L., and Chintala, S. (2015). Unsupervised representation
      learning with deep convolutional generative adversarial networks. ArXiv
      Prepr. ArXiv151106434.
```
