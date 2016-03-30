# Parallelized Stochastic Gradient Descent with Apache Spark

[![License](http://img.shields.io/:license-Apache%202-red.svg)](http://www.apache.org/licenses/LICENSE-2.0.txt)
[![Build Status](https://travis-ci.org/yu-iskw/spark-parallelized-sgd.svg?branch=master)](https://travis-ci.org/yu-iskw/spark-parallelized-sgd)
[![codecov.io](https://codecov.io/github/yu-iskw/spark-parallelized-sgd/coverage.svg?branch=master)](https://codecov.io/github/yu-iskw/spark-parallelized-sgd?branch=master)


This is an implementation of Parallelized Stochastic Gradient Descent with Apache Spark.
Since The current SGD in Mllib is not efficient, we should modify `org.apache.spark.mllib.optimization.GradientDiscent`.

Actually other parallel or distributed SGD methods have been sugested. However, almost of all them aren't fit for Spark.
For example, Downpour SGD or Asyncronous SGD require a parameter server. Therefore, it is a little annoying to implement them with Spark.

## Usage

```scala
import org.apache.spark.mllib.optimization._

val gradient = new LogisticGradient()
val updater = new SimpleSGDUpdater()

val dataRDD: RDD[(Double, Vector)] = ...
val (weights, loss) = new ParallelizedSGD(gradient, updater).optimize(dataRDD)

```

## Reference
Martin Zinkevich and Markus Weimer and Alexander J. Smola and Lihong Li, Parallelized Stochastic Gradient Descent
[http://research.microsoft.com/apps/pubs/?id=178845](http://research.microsoft.com/apps/pubs/?id=178845)
