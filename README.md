# Working neural network model templates in [pytorch][1].

This code was developed to help me turn neural network literature into practical
knowledge and techniques for using neural networks on actual problems. This repository was created to share the networks and notes I made while training the networks with others.

This repository contains:
---------
#### (1) Simple, self-contained, and working neural networks

Models are self contained, clearly coded and commented, and come with unit tests in an effort to provide a codebase that can be used
to quickly start using the models on your specific problem.

#### (2) Notes about what I learned while training the networks.

Literature is often missing tricks, caveats, and why hyperparameters were chosen. Notes on challenges and intricacies learned while training the networks are included for each model.

Due to limitations on embedding equations into markdown on GitHub, notes are published at [www.sciencesundires.com/sundries](http://www.sciencesundries.com/sundries)

Networks Implemented
---------

#### vanilla GANs

Linear and convolutional generative adversarial networks (GAN).
```
Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
    Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. “Generative
    Adversarial Nets.” In Advances in Neural Information Processing Systems,
    2672–2680.http://papers.nips.cc/paper/5423-generative-adversarial-nets.

Radford, A., Metz, L., and Chintala, S. (2015). Unsupervised representation
    learning with deep convolutional generative adversarial networks. ArXiv
    Prepr. ArXiv151106434.

```

To Install
---------
```
git clone git@github.com:cottersci/ModelTests.git
pipenv install
```
Training depends on my [pytorch_utils](https://github.com/cottersci/pytorch_utils) package, which is installed by pipenv. 

Other useful pytorch links
---------
- [pytorch.org][1]
- [pytorch.org/tutorials/][2]
- [The incredible pytorch][3]


[1]: http://pytorch.org/
[2]: http://pytorch.org/tutorials/
[3]: https://github.com/ritchieng/the-incredible-pytorch
[4]: http://yann.lecun.com/exdb/mnist/
[5]: http://ufldl.stanford.edu/housenumbers/
