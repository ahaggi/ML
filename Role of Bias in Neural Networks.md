
[Role of Bias in Neural Networks][0]

I think that biases are almost always helpful.  In effect, **a bias value allows you to shift the activation function to the left or right**, which may be critical for successful learning.

It might help to look at a simple example.  Consider this 1-input, 1-output network that has no bias:

![simple network][1]

The output of the network is computed by multiplying the input (x) by the weight (w<sub>0</sub>) and passing the result through some kind of activation function (e.g. a sigmoid function.)

Here is the function that this network computes, for various values of w<sub>0</sub>:

![network output, given different w0 weights][2]

Changing the weight w<sub>0</sub> essentially changes the "steepness" of the sigmoid.  That's useful, but what if you wanted the network to output 0 when x is 2?  Just changing the steepness of the sigmoid won't really work -- **you want to be able to shift the entire curve to the right**.

That's exactly what the bias allows you to do.  If we add a bias to that network, like so:

![simple network with a bias][3]

...then the output of the network becomes sig(w<sub>0</sub>*x + w<sub>1</sub>*1.0).  Here is what the output of the network looks like for various values of w<sub>1</sub>:

![network output, given different w1 weights][4]

Having a weight of -5 for w<sub>1</sub> shifts the curve to the right, which allows us to have a network that outputs 0 when x is 2.

  [0]: https://stackoverflow.com/a/2499936
  [1]: https://i.stack.imgur.com/bI2Tm.gif
  [2]: https://i.stack.imgur.com/ddyfr.png
  [3]: https://i.stack.imgur.com/oapHD.gif
  [4]: https://i.stack.imgur.com/t2mC3.png