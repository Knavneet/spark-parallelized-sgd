/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.optimization

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.norm

import org.apache.spark.Logging
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


/**
  * :: Experimental ::
  * Parallelized Stochastic Gradient Descent
  *
  * Martin Zinkevich and Markus Weimer and Alexander J. Smola and Lihong Li,
  * Parallelized Stochastic Gradient Descent
  * http://research.microsoft.com/apps/pubs/?id=178845
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */
@Experimental
class ParallelizedSGD private[spark] (
  private var gradient: Gradient,
  private var updater: SGDUpdater) extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
   * :: Experimental ::
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
   * Set the convergence tolerance. Default 0.001
   * convergenceTol is a condition which decides iteration termination.
   * The end of iteration is decided based on below logic.
   *
   *  - If the norm of the new solution vector is >1, the diff of solution vectors
   *    is compared to relative tolerance which means normalizing by the norm of
   *    the new solution vector.
   *  - If the norm of the new solution vector is <=1, the diff of solution vectors
   *    is compared to absolute tolerance which is not normalizing.
   *
   * Must be between 0.0 and 1.0 inclusively.
   */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: SGDUpdater): this.type = {
    this.updater = updater
    this
  }

  /**
   * :: DeveloperApi ::
   * Runs gradient descent on the given training data.
    *
    * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = ParallelizedSGD.runParallelizedSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol)
    weights
  }

}

/**
 * :: DeveloperApi ::
 * Top-level method to run parallelized stochastic gradient descent.
 */
@DeveloperApi
object ParallelizedSGD extends Logging {
  /**
   * Run parallelized stochastic gradient descent (SGD).
   *
   * @param data Input data for SGD. RDD of the set of data examples, each of
   *             the form (label, [feature values]).
   * @param gradient Gradient object (used to compute the gradient of the loss function of
   *                 one single data example)
   * @param updater Updater function to actually perform a gradient step in a given direction.
   * @param stepSize initial step size for the first step
   * @param numIterations number of iterations that SGD should be run.
   * @param regParam regularization parameter
   * @param miniBatchFraction fraction of the input data set that should be used for
   *                          one iteration of SGD. Default value 1.0.
   * @param convergenceTol Minibatch iteration will end before numIterations if the relative
   *                       difference between the current weight and the previous weight is less
   *                       than this value. In measuring convergence, L2 norm is calculated.
   *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */
  def runParallelizedSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: SGDUpdater,
      stepSize: Double,
      numIterations: Int,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Vector,
      convergenceTol: Double): (Vector, Array[Double]) = {

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
     * For the first iteration, the regVal will be initialized as sum of weight squares
     * if it's L2 updater; for L1 updater, the same logic is followed.
     */
    val status = updater.initStatus()
    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam, status)._2

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (avgWeights, avgRegVal, lossSum, batchSize) = data
        .sample(false, miniBatchFraction, 42 + i)
        .mapPartitions { iter =>
          var localStatus = updater.initStatus()
          var localWeights = bcWeights.value
          var oldWeights = bcWeights.value
          var localRegVal = 0.0
          var localLossSum = 0.0
          var count: Long = 0L
          var j = 1
          import scala.util.control.Breaks._
          breakable {
            iter.foreach { case (label, data) =>
              val (grad, loss) = gradient.compute(data, label, localWeights)
              val updated = updater.compute(localWeights, grad, stepSize, j, regParam, localStatus)
              localWeights = updated._1
              localRegVal = updated._2
              localStatus = updated._3
              localLossSum += loss
              count += 1L
              j += 1
              if (isConverged(oldWeights, localWeights, convergenceTol)) {
                break()
              }
              else {
                oldWeights = localWeights
              }
            }
          }
          Iterator.single((localWeights, localRegVal, localLossSum, count))
        }.treeReduce{ case ((w1, rv1, ls1, c1), (w2, rv2, ls2, c2)) =>
          val avgWeights =
            (w1.toBreeze * c1.toDouble + w2.toBreeze * c2.toDouble) / (c1 + c2).toDouble
          val avgRegVal = (rv1 * c1.toDouble + rv2 * c2.toDouble) / (c1 + c2).toDouble
          (Vectors.fromBreeze(avgWeights), avgRegVal, ls1 + ls2, c1 + c2)
        }

      if (batchSize > 0) {
        /**
         * lossSum is computed using the weights from the previous iteration
         * and regVal is the regularization value computed in the previous iteration as well.
         */
        val stochasticLoss = lossSum / batchSize + regVal
        stochasticLossHistory.append(stochasticLoss)
        log.warn(s"stochastic loss at step${i}: ${stochasticLoss}")
        weights = avgWeights
        regVal = avgRegVal

        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights.isDefined && currentWeights.isDefined) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)

  }

  /**
   * Alias of [[runParallelizedSGD]] with convergenceTol set to default value of 0.001.
   */
  def runParallelizedSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: SGDUpdater,
      stepSize: Double,
      numIterations: Int,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Vector): (Vector, Array[Double]) =
    ParallelizedSGD.runParallelizedSGD(data, gradient, updater, stepSize, numIterations,
                                    regParam, miniBatchFraction, initialWeights, 0.001)


  private def isConverged(
      previousWeights: Vector,
      currentWeights: Vector,
      convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.toBreeze.toDenseVector
    val currentBDV = currentWeights.toBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }
}
