const calc = require('./calc')






function Network(sizes) {
  this.numLayers = sizes.length
  // this.biases = [
  //   [[0.4393276076395669], [-0.8302987111077414], [-1.6919147197110789]],
  //   [[-0.39690947224808454]]
  // ]
  this.biases = [
    [[1], [2], [3]],
    [[4]]
  ]

  // for (var i = 1; i < sizes.length; i++)
  //   this.biases.push(calc.genMatrix(sizes[i][0], 1))  // size[n][0] becuase sizes = [ a ] where a is singeltonList [double], i.e. sizes=[[1.0],[2.3],[3.0]]

  // this.weights = [
  //   [[0.6421071385158644, 0.35911865812713367], [0.24565230602191543, 1.1423431038165999], [0.42488640822391427, -1.1744400323723452]],
  //   [[-2.1789378070935603, -0.05416367405528162, 1.445599312190188]]
  // ]
  this.weights = [
    [[1, 2], [3, 4], [5, 6]],
    [[7, 8, 9]]
  ]

  // for (let i = 1; i < sizes.length; i++)
  //   this.weights.push(calc.genMatrix(sizes[i][0], sizes[i - 1][0]))


  this.feedForward = (a) => {
    this.weights.map((w, indx) => {
      let b = this.biases[indx]
      let z = calc.addMat(calc.multiMat(w, a), b)
      a = calc.sigmoidVec(z)
    })
    return a // a(n+1)
  }

  this.sgd = (trainingData, epochs, miniBatchSize, eta, testData = null) => {
    // let n = trainingData.length
    for (let epc = 1; epc <= epochs; epc++) {

      calc.shuffle(trainingData)
      let miniBatchs = []

      for (let ind = 0; ind < trainingData.length; ind += miniBatchSize)
        miniBatchs.push(trainingData.slice(ind, ind + miniBatchSize))

      miniBatchs.map(miniBatch => {
        this.updateMiniBatch(miniBatch, eta)
      });
      console.log(`epoch nr ${epc} completed`)
    }

  }

  this.updateMiniBatch = (miniBatch, eta) => {
    // ∇ = nabla , eta = learning rate

    // biases,  nablaB, deltaNableB list of vectors
    // weights, nablaW, deltaNableW list of matrices
    nablaB = this.biases.map(vector => vector.map(val => 0))
    nablaW = this.weights.map(matrix => matrix.map(val => 0))

    miniBatch.map(({ input: x, output: y }) => {
      let { deltaNablaB, deltaNablaW } = backprop( x , y )
      nablaB = calc.addListOfMat(nablaB, deltaNablaB)
      nablaW = calc.addListOfMat(nablaW, deltaNablaW)
    })

    // avg the ∇biases and ∇weights
    // ∇C ≈ 1/m ∑ ∇Cx = [ ∂Cx/∂w , ∂Cx/∂b ]
    // w=w−η⋅∇C(w)
    // b=b−η⋅∇C(b)
    nablaB = calc.divideMatNum(nablaB, eta/ miniBatchs.length)
    nablaW = calc.divideMatNum(nablaW, eta/ miniBatchs.length)

    this.biases  = calc.subtractListOfMat(this.biases, nablaB)
    this.weights = calc.subtractListOfMat(this.weights, nablaW)

  }

  this.backProp = (x, y) => {
    nablaB = this.biases.map(row => row.map(val => 0))
    nablaW = this.weights.map(row => row.map(val => 0))

    //OBS sjekk om formen på vektor x og y er rett!!
    let activation = x
    let activations = [x]

    zs = []

    this.weights.map((w, indx) => {
      let b = this.biases[indx]

      let z = calc.addMat(calc.multiMat(w, activation), b)
      zs.push(z)

      activation = calc.sigmoidVec(z)
      activations.push(activation)
    })

    //  backward pass: finding ∇w and ∇b for the "output layer", is similar to that in "single-layer networks"
    //                                          https://blog.yani.io/backpropagation/
    // ∇C =  [ ∂Cx/∂w , ∂Cx/∂b ]
    // nablaW = ∂Cx/∂w = ∂z(L)/∂w(L) * ∂a(L)/∂z(L) * ∂C(L)/∂a(L)
    // nablaB = ∂Cx/∂b = ∂z(L)/∂b(L) * ∂a(L)/∂z(L) * ∂C(L)/∂a(L)
    // 
    // nablaW = ∂Cx/∂w =    a(L-1)   *   σ'(z(L))  * 2 (a(L) - y)
    // nablaB = ∂Cx/∂b =      1      * σ' (z(L))   * 2 (a(L) - y)
    // 
    // delta = σ'(z(L))  * 2 (a(L) - y)
    // nablaW = ∂Cx/∂w =    a(L-1)   *   delta
    // nablaB = ∂Cx/∂b =      1      *   delta
    // 
    // Notice that the 2's are omitted here and 
    // when we avrage the ∇biases and ∇weights in "this.updateMiniBatch(..)"

    let aLast = activations[activations.length - 1]
    let zLast = zs[zs.length - 1]

    // alt use https://mathjs.org/docs/core/chaining.html
    // δ = (a - y) ⊙ σ'(z) for the output layer
    let a1 = calc.subtractMat(aLast, y) // (a-y) is the derivative of (a - y)^2, "the 2's are omitted" if it was (y-a)^2 the derivative will be -(y-a)
    let a2= calc.sigmoidPrime(zLast)
    let delta = calc.hadamard(a1,a2 )


    // nablaW = ∂Cx/∂w =    a(L-1)   *   delta
    // nablaB = ∂Cx/∂b =      1      *   delta
    let activationPrevLayerT = calc.transpose (activations[activations.length - 2])
    nablaB[nablaB.length - 1] = delta
    nablaW[nablaW.length - 1] = calc.multiMat(delta, activationPrevLayerT) // activationPrevLayerT: (delta(vector)) . (activationPrevLayer(vector)T) = matrix 

    for (let indx = this.numLayers - 1 ; indx > 1; indx--) {
      // if the iteration is for L1 
      //    To calculate δ L1 we need the flwg
      //        z[0] av zs = [ L1 , L2 ] ==> L1
      //        w[1] av w  = [ L1 , L2 ] ==> L2
      //        δ L2

      let z = zs[zs.length-indx] 
      // δ(l)=( (w(l+1))T * δ(l+1) ) ⊙ σ′(z(l))

      wNextLayerT  = calc.transpose(this.weights[weights.length - indx + 1]) 
      deltaNextLayer = delta
      let sp = calc.sigmoidPrime(z)

      let temp1 = calc.multiMat(wNextLayerT , deltaNextLayer) //  [ [7],[8],[9] ] * [[12]] 
      delta = calc.hadamard(temp1 , sp)

    //    To calculate ∇W L1 we need the flwg
    //        δ L1
    //        a[0] av a = [ L0 , L1 , L2 ] ==> L0
    // nablaW = ∂Cx/∂w =    a(L-1)   *   delta
    // nablaB = ∂Cx/∂b =      1      *   delta
      nablaB[nablaB.length - indx] = delta
      activationPrevLayerT = calc.transpose (activations[activations.length - indx -1])
      nablaW[nablaW.length - indx] = calc.multiMat(delta , activationPrevLayerT)
    }
    return { nablaB: nablaB, nablaW: nablaW }
  }

}//Network


module.exports = { Network }