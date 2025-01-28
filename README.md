# Local SGD vs. Minibatch SGD: A Study

Local SGD and Minibatch SGD are two widely used optimization algorithms in machine learning and other tasks requiring parallel computation. Despite their popularity, developing a theoretical understanding of their limitations and properties remains challenging, even in smooth convex cases.

This project is based on the paper ["Is Local SGD Better than Minibatch SGD?"](https://arxiv.org/pdf/2002.07839) by Woodworth et al. (2020). The authors aimed to bridge the gap in the theoretical understanding of the Local SGD algorithm by highlighting its advantages and drawbacks compared to the well-studied Minibatch SGD algorithm. They also identified the regimes in which each method performs optimally.

To validate the findings and explore the impact of key parameters, several numerical experiments were conducted based on the results presented in the paper.
