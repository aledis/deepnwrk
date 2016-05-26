package com.aledis.ml.deep

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.Random

/**
  * Created by sasha on 5/19/16.
  */
class Network(val layerSizes:List[Int]) {
  type Biases = DenseVector[Double]
  type Weights = DenseMatrix[Double]
  type Activation = DenseVector[Double]
  type Data = List[(Activation,Activation)]

  var biases:List[Biases] = layerSizes.drop(1).map(size => DenseVector.rand[Double](size))
  var weights:List[Weights] = layerSizes.drop(1).zip(layerSizes.dropRight(1)).map({case (rows,cols) => DenseMatrix.rand(rows,cols) })

  private def sigmoid(z:Activation):Activation =  {
    z.map(e => 1/(1+Math.exp(-e)))
  }

  private def sigmoidPrime(z:Activation):Activation = {
    val sigmoidVal = sigmoid(z)
    sigmoidVal*(1-sigmoidVal)
  }

  private def costDerivative(output:Activation, y:Activation):Activation = output - y

  private def getNablas():(List[Biases],List[Weights]) =
    (biases.map(bias => DenseVector.zeros[Double](bias.length)),
      weights.map(weight => DenseMatrix.zeros[Double](weight.rows,weight.cols)))


  def feedForward(a:Activation):Activation = {
    biases.zip(weights).foldRight(a)({case (b,w) => (w dot a) + b})
  }

  def SGD(trainingData:Data, numEpochs:Int, miniBatchSize:Int, eta:Double, testData:Data) = {
    (1 to numEpochs).foreach(epoch => Random.shuffle(trainingData)
      .grouped(miniBatchSize)
      .foreach(mBatch=>updateMiniBatch(mBatch, eta)))
  }

  private def updateMiniBatch(miniBatch:Data, eta:Double) = {
    var (nabla_b, nabla_w) = getNablas()
    miniBatch.foreach({case (input,output) =>
        val (d_nabla_b, d_nabla_w) = backProp(input, output)
        nabla_b = nabla_b + d_nabla_b
        nabla_w = nabla_w + d_nabla_w
    })
    weights = weights.zip(nabla_w).map({case (w,nw) => w - (eta/miniBatch.length)*nw})
    biases = biases.zip(nabla_b).map({case (b,nb) => b - (eta/miniBatch.length)*nb})
  }

  private def backProp(input:Activation, output:Activation):(List[Biases], List[Weights]) = {
    var ( nabla_b, nabla_w) = getNablas()
    var activation = input
    var activations = List[Activation](activation)
    var zs = List[Activation]()

    biases.zip(weights).foreach({case (b,w) =>
        val z = w dot activation + b
        zs = zs :+ z
        activation = sigmoid(z)
        activations = activations :+ activation
    })
    var delta:Activation = costDerivative(activation,output) * sigmoidPrime(zs.last)
    nabla_b = nabla_b.updated(nabla_b.length-1,delta)
    nabla_w = nabla_w.updated(nabla_w.length-1,delta dot activations(activations.length-2).t)
    (2 to layerSizes.length).foreach(l => {
      val z = zs(zs.length - l)
      val sp = sigmoidPrime(z)
      val dotP:Activation = (weights(weights.length-l+1).t dot delta)
      delta = dotP*sp
      nabla_b = nabla_b.updated(nabla_b.length -l, delta)
      nabla_w = nabla_w.updated(nabla_w.length -l, activations(activations.length - l -1).t)
    }
    )
    (nabla_b,nabla_w)
  }
}
