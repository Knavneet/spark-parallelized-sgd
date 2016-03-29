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

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.{RegressionMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.ParallelizedSGD
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.{SparkContext, SparkFunSuite}
import org.apache.spark.mllib.util.{LinearDataGenerator, MLlibTestSparkContext}
import org.apache.spark.rdd.RDD

class LinearRegressionWithPSGDSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("run") {

    val rnd = new XORShiftRandom()
    val dim = 10
    val intercept = 0.0
    val weights = (1 to dim).map(i => rnd.nextGaussian()).toArray
    val xMean = Array.fill(dim)(0.0)
    val xVariance = Array.fill(dim)(1.0 / 3.0)
    val nPoints = 10000
    val seed = 1
    val eps = 1e-4
    val sparsity = 0.5
    val trainData = sc.parallelize(
      LinearDataGenerator.generateLinearInput(
        intercept, weights, xMean, xVariance, nPoints, seed, eps, sparsity), 2)
    val testData = sc.parallelize(
      LinearDataGenerator.generateLinearInput(
        intercept, weights, xMean, xVariance, nPoints, seed, eps, sparsity), 2)
    val lr = new LinearRegressionWithPSGD()
    val model = lr.run(trainData)
    val scoreAndLabel = testData.map { lp =>
      val trueLabel = lp.label
      val predictedLabel = model.predict(lp.features)
      (trueLabel, predictedLabel)
    }
    val metrics = new RegressionMetrics(scoreAndLabel)
    val mse = metrics.meanSquaredError
    val rmse = metrics.rootMeanSquaredError
    1
  }
}

