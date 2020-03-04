const math = require('mathjs')

const genMatrix = (m, n) => {
    var matrix = []
    for (var i = 0; i < m; i++) {
        matrix.push([])
        for (var j = 0; j < n; j++)
            matrix[i][j] = genRandom()
    }
    return matrix
}

const genRandom = () => {
    // produce pseudo-random number sampling method for generating pairs of independent, standard, normally distributed N(0, 1) random numbers
    // boxMuller transform
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

const addMat = (m1_, m2_) => {
    let m1 = math.matrix(m1_)
    let m2 = math.matrix(m2_)
    return math.add(m1, m2)._data
}
const addListOfMat = (m1s_, m2s_) => {
    if (m1s_.length !== m2s_.length)
        return null
    return m1s_.map((_, ind) =>
        addMat(m1s_[ind], m2s_[ind]))
}

const subtractMat = (m1_, m2_) => {
    let m1 = math.matrix(m1_)
    let m2 = math.matrix(m2_)
    return math.subtract(m1, m2)._data
}

const subtractListOfMat = (m1s_, m2s_) => {
    if (m1s_.length !== m2s_.length)
        return null
    return m1s_.map((_, ind) =>
        subtractMat(m1s_[ind], m2s_[ind]))
}



const multiMat = (m1_, m2_) => {
    let m1 = math.matrix(m1_)
    let m2 = math.matrix(m2_)
    return math.multiply(m1, m2)._data
}

const multiListOfMat = (m1s_, m2s_) => {
    if (m1s_.length !== m2s_.length)
        return null
    return m1s_.map((_, ind) =>
        multiMat(m1s_[ind], m2s_[ind]))
}

const multiMatNum = (m1_, num) => {
    let m1 = math.matrix(m1_)
    return math.multiply(m1, num)._data
}


const divideMatNum = (m1_, num) => {
    let m1 = math.matrix(m1_)
    return math.divide(m1, num)._data
}

const transpose = (m1_) => math.transpose(m1_)

const hadamard = (m1_, m2_) => {
    let m1 = math.matrix(m1_)
    let m2 = math.matrix(m2_)
    return math.dotMultiply(m1, m2)._data

}


const sigmoid = (t) => 1 / (1 + Math.pow(Math.E, -t));

// v is [ [double] ]
const sigmoidVec = (v) => v.map(arry => arry.map(sigmoid))

const sigmoidPrime = (v) => {
    // sigmoidVec(v) * (1 - sigmoidVec(v))
    let v1 = sigmoidVec(v)
    let v2 = v.map(arry => arry.map(num => 1 - sigmoid(num)))
    return math.dotMultiply(v1, v2)
}

// retruns random value for comparing 2 elem in the array
const shuffle = (array) => array.sort(() => Math.random() - 0.5);

const shuffle_ = (array) => {
    var currentIndex = array.length;
    var temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {
        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }
    return array;
}


module.exports =
    {
        genMatrix,
        addMat,
        addListOfMat,
        subtractMat,
        subtractListOfMat,
        multiMat,
        multiListOfMat,
        divideMatNum,
        multiMatNum,
        transpose,
        hadamard,
        sigmoid,
        sigmoidVec,
        sigmoidPrime,
        shuffle,
        shuffle_
    }