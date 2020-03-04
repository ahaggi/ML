const express = require('express');
const app = express();
const path = require('path');
const { createCanvas } = require('canvas')
var mnist = require('mnist'); // this line is not needed in the browser
const math = require('mathjs')
const calc = require('./calc')
const n = require('./network')

var set = mnist.set(8000, 10);

var trainingSet = set.training;
var testSet = set.test;


let draw_ = async (req, res) => {
  const canvas = createCanvas(200, 200)
  const ctx = canvas.getContext('2d')

  var digit = mnist[3].get();
  // console.log(testSet)

  mnist.draw(digit, ctx); // draws a '1' mnist digit in the canvas
  const buf2 = canvas.toBuffer('image/png', { compressionLevel: 3, filters: canvas.PNG_FILTER_NONE })
  res.send(buf2)
}


app.get('/image', draw_)


app.get('/', (req, res) => {

  let sizes = [[2], [3], [1]];

  let t = new n.Network(sizes)
  t.backProp([[10],[20]] , [[1]])
  // let a = calc.multiMat( [[2]] , math.transpose([[2 , 4]]) )

  // console.log( a )
  res.sendFile( path.join(__dirname, 'index.html') );
})

var server = app.listen( 5000 )