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

import scala.math._

import breeze.linalg.{axpy => brzAxpy, norm => brzNorm, Vector => BV}

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * :: DeveloperApi ::
  * Class used to perform steps (weight update) using Gradient Descent methods.
  *
  * For general minimization problems, or for regularized problems of the form
  *         min  L(w) + regParam * R(w),
  * the compute function performs the actual update step, when given some
  * (e.g. stochastic) gradient direction for the loss L(w),
  * and a desired step-size (learning rate).
  *
  * The updater is responsible to also perform the update coming from the
  * regularization term R(w) (if any regularization is used).
  */
@DeveloperApi
abstract class SGDUpdater extends Serializable {
  // Type for a status to target updater
  type Status

  /**
    * Compute an updated value for weights given the gradient, stepSize, iteration number and
    * regularization parameter. Also returns the regularization value regParam * R(w)
    * computed using the *updated* weights.
    *
    * @param weightsOld - Column matrix of size dx1 where d is the number of features.
    * @param gradient - Column matrix of size dx1 where d is the number of features.
    * @param stepSize - step size across iterations
    * @param iter - Iteration number
    * @param regParam - Regularization parameter
    * @param status - Updater status
    * @return A tuple of e elements. The first element is a column matrix containing updated weights,
    *         and the second element is the regularization value computed using updated weights.
    *         The third element is the updated status.
    */
  def compute(
    weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double,
    status: Status): (Vector, Double, Status)

  // Gets an initialized status of a updater.
  def initStatus(): Status

  // Class for a status of updater
  abstract class UpdaterStatus extends Serializable
}

/**
  * :: DeveloperApi ::
  * A simple updater for gradient descent *without* any regularization.
  * Uses a step-size decreasing with the square root of the number of iterations.
  */
@DeveloperApi
class SimpleSGDUpdater extends SGDUpdater {
  override type Status = Unit

  override def initStatus(): Status = ()

  override def compute(
    weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double,
    status: Status): (Vector, Double, Status) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)

    (Vectors.fromBreeze(brzWeights), 0, status)
  }
}

/**
  * :: DeveloperApi ::
  * Updater for L1 regularized problems.
  *          R(w) = ||w||_1
  * Uses a step-size decreasing with the square root of the number of iterations.

  * Instead of subgradient of the regularizer, the proximal operator for the
  * L1 regularization is applied after the gradient step. This is known to
  * result in better sparsity of the intermediate solution.
  *
  * The corresponding proximal operator for the L1 norm is the soft-thresholding
  * function. That is, each weight component is shrunk towards 0 by shrinkageVal.
  *
  * If w >  shrinkageVal, set weight component to w-shrinkageVal.
  * If w < -shrinkageVal, set weight component to w+shrinkageVal.
  * If -shrinkageVal < w < shrinkageVal, set weight component to 0.
  *
  * Equivalently, set weight component to signum(w) * max(0.0, abs(w) - shrinkageVal)
  */
@DeveloperApi
class L1SGDUpdater extends SGDUpdater {
  override type Status = Unit

  override def initStatus(): Status = ()

  override def compute(
    weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double,
    status: Status): (Vector, Double, Status) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    // Take gradient step
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regParam * thisIterStepSize
    var i = 0
    val len = brzWeights.length
    while (i < len) {
      val wi = brzWeights(i)
      brzWeights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
      i += 1
    }

    (Vectors.fromBreeze(brzWeights), brzNorm(brzWeights, 1.0) * regParam, status)
  }
}

/**
  * :: DeveloperApi ::
  * Updater for L2 regularized problems.
  *          R(w) = 1/2 ||w||^2
  * Uses a step-size decreasing with the square root of the number of iterations.
  */
@DeveloperApi
class SquaredL2SGDUpdater extends SGDUpdater {
  override type Status = Unit

  override def initStatus(): Status = ()

  override def compute(
    weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double,
    status: Status): (Vector, Double, Status) = {
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    val norm = brzNorm(brzWeights, 2.0)

    (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm, status)
  }
}

/**
  * :: DeveloperApi ::
  * Updater for AdaGrad.
  *          r = r + grad * grad
  *          w = w_old - lr * grad / (sqrt(r) + eps)
  * Uses a step-size decreasing with the square root of the number of iterations.
  *
  * https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad
  */
@DeveloperApi
class AdaGradSGDUpdatre extends SGDUpdater {
  override type Status = AdaGradUpdaterStatus

  override def initStatus(): Status = new Status()

  override def compute(
    weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double,
    status: Status): (Vector, Double, Status) = {
    val updatedStatus = status.update(gradient)
    val accumSquaredGrad = updatedStatus.accumSquaredGradient.get
    val g = gradient.toBreeze / (breeze.numerics.pow(accumSquaredGrad.toBreeze + 1.0, 0.5))

    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, g, brzWeights)

    (Vectors.fromBreeze(brzWeights), 0.0, updatedStatus)
  }
  class AdaGradUpdaterStatus(val accumSquaredGradient: Option[Vector]) extends UpdaterStatus {
    def this() = this(None)

    def update(gradient: Vector): AdaGradUpdaterStatus = {
      val squaredGrad = gradient.toBreeze :* gradient.toBreeze
      val updated = accumSquaredGradient match {
        case None => squaredGrad
        case _ => accumSquaredGradient.get.toBreeze + squaredGrad
      }
      new AdaGradUpdaterStatus(Some(Vectors.fromBreeze(updated)))
    }
  }
}

/**
  * :: DeveloperApi ::
  * Updater for Adam.
  *          v <- beta * v - (1 - beta) * grad
  *          r <- gamma * r - (1 - gamma) * grad * grad
  *          w <- w - (learning_rate / (sqrt(r / 1 - r ^ t) + eps)) * (v / (1 - beta ^ t))
  * Uses a step-size decreasing with the square root of the number of iterations.
  *
  * @param eps epsilon
  */
@DeveloperApi
class AdamSGDUpdater(val beta: Double, val gamma: Double, val eps: Double)
  extends SGDUpdater {

  def this() = this(0.9, 0.999, 1e-8)

  override type Status = AdamUpdaterStatus

  override def initStatus(): Status = {
    new AdamUpdaterStatus()
  }

  override def compute(weightsOld: Vector,
    gradient: Vector,
    stepSize: Double,
    iter: Int,
    regParam: Double,
    status: Status): (Vector, Double, Status) = {
    val updatedStatus = status.update(gradient, beta, gamma)
    val (v, r) = (updatedStatus.v.get, updatedStatus.r.get)
    val fix1 = breeze.numerics.pow(1.0 - breeze.numerics.pow(r.toBreeze, iter.toDouble), 0.5) + eps
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val learningRage = thisIterStepSize / (1.0 - math.pow(beta, iter))
    val g = v.toBreeze / fix1
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-learningRage, g, brzWeights)

    (Vectors.fromBreeze(brzWeights), 0.0, updatedStatus)
  }

  class AdamUpdaterStatus(val v: Option[Vector], val r: Option[Vector]) extends UpdaterStatus {
    def this() = this(None, None)

    def update(gradient: Vector, beta: Double, gamma: Double): AdamUpdaterStatus = {
      val squaredGrad = gradient.toBreeze :* gradient.toBreeze
      val updated = v.isEmpty && r.isEmpty match {
        case true => (gradient.toBreeze * (1 - beta), squaredGrad * (1 - gamma))
        case false => (
          this.v.get.toBreeze * beta + gradient.toBreeze * (1 - beta),
          this.r.get.toBreeze * gamma + squaredGrad * (1 - gamma))
      }
      new AdamUpdaterStatus(
        Some(Vectors.fromBreeze(updated._1)),
        Some(Vectors.fromBreeze(updated._2)))
    }
  }
}

