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

package org.apache.spark.mllib.regression

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._

class LinearRegressionWithPSGD private[mllib] (
    private var stepSize: Double,
    private var numIterations: Int,
    private var miniBatchFraction: Double
  ) extends GeneralizedLinearAlgorithm[LinearRegressionModel] with Serializable {

  def this() = this(1.0, 100, 1.0)

  private val gradient = new LeastSquaresGradient()
  private val updater = new SimpleSGDUpdater()
  override val optimizer: Optimizer = new ParallelizedSGD(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setMiniBatchFraction(miniBatchFraction)

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new LinearRegressionModel(weights, intercept)
  }
}
