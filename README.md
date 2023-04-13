# svm_lab
svm lab
## Work1

![1681421250033](https://user-images.githubusercontent.com/51303014/231888604-c28dbeaa-87e1-4892-a19e-c2135664c22b.jpg)

sv num: 11

train error: 0.02

test error: 0

## Work2

In my problem, when C = 1, the test sample error is already 0, but the training sample error must increase C to about 190 to become 0. The higher C, the lower the model tolerance and the easier it is for the model to overfit. Therefore, it is not necessary to raise C to reduce the error to 0.  

## Work3

The best kernel is poly kernel with degree = 2 and linear kernel. They both have 0 error on test data.

## Work4

Same as Work3.

## Work5

![1681424432973](https://user-images.githubusercontent.com/51303014/231896008-e2a78b95-8c9f-4678-a17a-79ace919b495.jpg)

![1681424448492](https://user-images.githubusercontent.com/51303014/231896024-de98d5e4-f8bf-482e-a492-f467b0b239d3.jpg)

![1681424464453](https://user-images.githubusercontent.com/51303014/231896036-51f34441-ff31-434d-8860-6aaa928f8f70.jpg)

I raised the gamma from 0.1 to 10. According to the picture, it can be seen that as the gamma value increases, the number of support vectors gradually decreases, and the division of the entire region becomes strict at the boundary, which is the so-called overfitting.

## Work6

![1681427055269](https://user-images.githubusercontent.com/51303014/231901537-a1a25bb5-0286-45eb-8f8c-8bfc900a4797.jpg)

As epsilon grows, so does mse. Epsilon represents the width of the allowed support vector. The smaller the epsilon, the more points are outside, which will result in a greater penalty or even overfitting. If the epsilon is large, the smoother the line is, the less points are penalized until it becomes even a straight line. So when we use epsilon-svr, we need to weigh the two situations to fit a good result.
