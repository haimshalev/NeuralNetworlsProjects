using System;
using System.Collections.Generic;

namespace ClassifyHebLettersUsingBackProp
{
    public class NeuralNetwork
    {
        #region Properties

        private static Random _rnd;

        private readonly int _numInput;
        private readonly int _numHidden;
        private readonly int _numOutput;

        private readonly double[] _inputs;

        private readonly double[][] _ihWeights; // input-hidden
        private readonly double[] _hBiases;
        private readonly double[] _hOutputs;

        private readonly double[][] _hoWeights; // hidden-output
        private readonly double[] _oBiases;

        private readonly double[] _outputs;

        // back-prop specific arrays (these could be local to method UpdateWeights)
        private readonly double[] _oGrads; // output gradients for back-propagation
        private readonly double[] _hGrads; // hidden gradients for back-propagation

        // back-prop momentum specific arrays (these could be local to method Train)
        private readonly double[][] _ihPrevWeightsDelta;  // for momentum with back-propagation
        private readonly double[] _hPrevBiasesDelta;
        private readonly double[][] _hoPrevWeightsDelta;
        private readonly double[] _oPrevBiasesDelta; 

        #endregion

        /// <summary>
        /// Cto'r
        /// </summary>
        /// <param name="numInput">number of input neurons</param>
        /// <param name="numHidden">number of hidden layer neurons</param>
        /// <param name="numOutput">number of output layer neurons, used also for labeling</param>
        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            // for InitializeWeights() and Shuffle()
            _rnd = new Random(0); 

            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;

            _inputs = new double[numInput];

            _ihWeights = MakeMatrix(numInput, numHidden);
            _hBiases = new double[numHidden];
            _hOutputs = new double[numHidden];

            _hoWeights = MakeMatrix(numHidden, numOutput);
            _oBiases = new double[numOutput];

            _outputs = new double[numOutput];

            // back-prop related arrays below
            _hGrads = new double[numHidden];
            _oGrads = new double[numOutput];

            _ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            _hPrevBiasesDelta = new double[numHidden];
            _hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            _oPrevBiasesDelta = new double[numOutput];

            // initialize the weights randomaly
            InitializeWeights();
        } 

        /// <summary>
        /// sets the internal weights accordingly to the weights vector
        /// </summary>
        /// <param name="weights">the desired weights vector</param>
        private void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            var numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            var k = 0; // points into weights param

            for (var i = 0; i < _numInput; ++i)
                for (var j = 0; j < _numHidden; ++j)
                    _ihWeights[i][j] = weights[k++];
            for (var i = 0; i < _numHidden; ++i)
                _hBiases[i] = weights[k++];
            for (var i = 0; i < _numHidden; ++i)
                for (var j = 0; j < _numOutput; ++j)
                    _hoWeights[i][j] = weights[k++];
            for (var i = 0; i < _numOutput; ++i)
                _oBiases[i] = weights[k++];
        }

        /// <summary>
        /// initialize the weights and biases to small random values between -0.01 to 0.01
        /// </summary>
        private void InitializeWeights()
        {
            var numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            var initialWeights = new double[numWeights];
            const double lo = -0.01;
            const double hi = 0.01;
            for (var i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * _rnd.NextDouble() + lo;
            SetWeights(initialWeights);
        }

        /// <summary>
        /// Returns the current set of weights
        /// </summary>
        /// <returns>the current network's weights</returns>
        public double[] GetWeights()
        {
            var numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            var result = new double[numWeights];
            var wightIndex = 0;

            foreach (var inputHiddenWeight in _ihWeights)
                for (var j = 0; j < _ihWeights[0].Length; ++j)
                    result[wightIndex++] = inputHiddenWeight[j];
            foreach (var hiddenBias in _hBiases)
                result[wightIndex++] = hiddenBias;
            foreach (var hiddenOutputWeight in _hoWeights)
                for (var j = 0; j < _hoWeights[0].Length; ++j)
                    result[wightIndex++] = hiddenOutputWeight[j];
            foreach (var outputBias in _oBiases)
                result[wightIndex++] = outputBias;
            return result;
        }

        /// <summary>
        /// return the output values of the last computation
        /// </summary>
        /// <returns>the output array of the last network computation</returns>
        public double[] GetOutputs()
        {
            // copy the internal output array and return it
            var retResult = new double[_numOutput];
            Array.Copy(_outputs, retResult, retResult.Length);
            return retResult;
        }

        /// <summary>
        /// Feed forward the network and return it's output
        /// It also update the inner properties of the network with the latest neuro values 
        /// </summary>
        /// <param name="inputValues">the input vector, should be from the size of the input neurons</param>
        /// <returns>the output of the network for the input vector</returns>
        private double[] FeedForward(double[] inputValues)
        {
            if (inputValues.Length != _numInput)
                throw new Exception("Bad inputValues array length");

            // hidden nodes sums scratch array
            var hSums = new double[_numHidden]; 

            // output nodes sums
            var oSums = new double[_numOutput]; 

            // copy x-values to inputs
            for (var i = 0; i < inputValues.Length; ++i) 
                _inputs[i] = inputValues[i];

            // compute i-h sum of weights * inputs
            for (var j = 0; j < _numHidden; ++j)  
                for (var i = 0; i < _numInput; ++i)
                    hSums[j] += _inputs[i] * _ihWeights[i][j]; 
            
            // add biases to input-to-hidden sums
            for (var i = 0; i < _numHidden; ++i)  
                hSums[i] += _hBiases[i];

            // apply the activation function to the hidden layer
            for (var i = 0; i < _numHidden; ++i) 
            {
                // was HyperTan function for the activation of the hidden layert
                //_hOutputs[i] = HyperTanFunction(hSums[i]); 

                _hOutputs[i] = SigmoidFunction(hSums[i]);

            }

            // compute h-o sum of weights * hOutputs
            for (var j = 0; j < _numOutput; ++j)   
                for (var i = 0; i < _numHidden; ++i)
                    oSums[j] += _hOutputs[i] * _hoWeights[i][j];
            
            // add biases to input-to-hidden sums
            for (var i = 0; i < _numOutput; ++i)  
                oSums[i] += _oBiases[i];

            // apply the activation function to the output layer
            
            // was softMax activation function
            //var softOut = SoftmaxFunction(oSums); // softmax activation does all outputs at once for efficiency
            //Array.Copy(softOut, _outputs, softOut.Length);
            
            for (var i = 0; i < _numOutput; ++i)
            {
                _outputs[i] = SigmoidFunction(oSums[i]);
            }

            // return the outputs
            return GetOutputs();
        }

        /// <summary>
        /// update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
        /// assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and  matrices have values (other than 0.0)
        /// </summary>
        /// <param name="tValues">true output values</param>
        /// <param name="learnRate">the back propagation learning rate</param>
        /// <param name="momentum">the back propagation momentum const</param>
        private void UpdateWeights(double[] tValues, double learnRate, double momentum)
        {
            
            if (tValues.Length != _numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. compute output gradients
            for (var i = 0; i < _oGrads.Length; ++i)
                _oGrads[i] = SigmoidFunctionDerviative(_outputs[i], true) * (tValues[i] - _outputs[i]);

            // 2. compute hidden gradients
            for (var i = 0; i < _hGrads.Length; ++i)
            {
                var sum = 0.0;

                // each hidden delta is the sum of numOutput terms
                for (var j = 0; j < _numOutput; ++j) 
                {
                    var x = _oGrads[j] * _hoWeights[i][j];
                    sum += x;
                }
                _hGrads[i] = SigmoidFunctionDerviative(_hOutputs[i], true) * sum;
            }

            // 3a. update hidden weights (gradients must be computed right-to-left but weights
            // can be updated in any order)
            for (var i = 0; i < _ihWeights.Length; ++i) // 0..2 (3)
            {
                for (var j = 0; j < _ihWeights[0].Length; ++j) // 0..3 (4)
                {
                    var delta = learnRate * _hGrads[j] * _inputs[i]; // compute the new delta
                    _ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                    // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    _ihWeights[i][j] += momentum * _ihPrevWeightsDelta[i][j];
                    // weight decay would go here
                    _ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
                }
            }

            // 3b. update hidden biases
            for (var i = 0; i < _hBiases.Length; ++i)
            {
                // the 1.0 below is the constant input for any bias; could leave out
                var delta = learnRate * _hGrads[i] * 1.0;
                _hBiases[i] += delta;
                _hBiases[i] += momentum * _hPrevBiasesDelta[i]; // momentum
                // weight decay here
                _hPrevBiasesDelta[i] = delta; // don't forget to save the delta
            }

            // 4. update hidden-output weights
            for (var i = 0; i < _hoWeights.Length; ++i)
            {
                for (var j = 0; j < _hoWeights[0].Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    var delta = learnRate * _oGrads[j] * _hOutputs[i];
                    _hoWeights[i][j] += delta;
                    _hoWeights[i][j] += momentum * _hoPrevWeightsDelta[i][j]; // momentum
                    // weight decay here
                    _hoPrevWeightsDelta[i][j] = delta; // save
                }
            }

            // 4b. update output biases
            for (var i = 0; i < _oBiases.Length; ++i)
            {
                var delta = learnRate * _oGrads[i] * 1.0;
                _oBiases[i] += delta;
                _oBiases[i] += momentum * _oPrevBiasesDelta[i]; // momentum
                // weight decay here
                _oPrevBiasesDelta[i] = delta; // save
            }
        }

        /// <summary>
        /// train a back-prop style NN classifier using learning rate and momentum no weight decay
        /// </summary>
        /// <param name="trainData">the training set</param>
        /// <param name="maxEpochs">max num of training iterations (on each epoch we going over all the data)</param>
        /// <param name="learnRate">the back propagation learning rate</param>
        /// <param name="momentum">the back propagation mumentum const</param>
        /// <param name="mseLimit">the desired summed mse for all the data</param>
        public void Train(List<InputDataStructure> trainData, int maxEpochs, double learnRate, double momentum, double mseLimit)
        {
            var epoch = 0;

            // create an indexes array
            var sequence = new int[trainData.Count];
            for (var i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            // while we didn't get to the epoch limit
            while (epoch < maxEpochs)
            {
                Console.WriteLine("Starting epcoch number " + epoch);

                // calculate meanSquaredError on the training set to see that we are not over fitting
                var mse = MeanSquaredError(trainData);

                // if we got to the desired mse, stop training
                if (mse < mseLimit) break;

                // visit each training data in random order
                Shuffle(sequence); 

                // run over all the training data and update weights 
                for (var i = 0; i < trainData.Count; ++i)
                {
                    var idx = sequence[i];

                    // extract the input vector
                    double[] xValues = trainData[idx].GetDataVectorCopy();

                    // extract the target vector
                    double[] tValues = trainData[idx].GetTargetVectorCopy();

                    // copy xValues in, compute outputs (and store them in the internal members)
                    FeedForward(xValues);

                    // use back-prop to find better weights using the stored values in the internal members
                    UpdateWeights(tValues, learnRate, momentum); 
                } 

                // increase the epoch number
                ++epoch;
            }
        }

        /// <summary>
        /// calculate the averaged meanSquaredError of the output of the NN classifier on the given labeld data
        /// used as a training stopping condition
        /// </summary>
        /// <param name="trainData">input labeld data</param>
        /// <returns>averaged mean squared error</returns>
        private double MeanSquaredError(List<InputDataStructure> trainData) 
        {
            var sumSquaredError = 0.0;

            // for each vector in the input db
            foreach (var inputData in trainData)
            {
                // extract the input vector
                double[] xValues = inputData.GetDataVectorCopy();

                // extract the target vector
                double[] tValues = inputData.GetTargetVectorCopy();

                // compute output using current weights
                var yValues = FeedForward(xValues); 

                // calculate the mean squared error and add it to the previous ones
                for (var j = 0; j < _numOutput; ++j)
                    sumSquaredError += (tValues[j] - yValues[j]) * (tValues[j] - yValues[j]);
            }

            // return the averaged mean squared error
            return sumSquaredError / trainData.Count;
        }

        /// <summary>
        /// calculate the accurracy of the network on the given test set
        /// </summary>
        /// <param name="testData">the input test set</param>
        /// <param name="printEachLetterSummary">if true , prints each letter result and target</param>
        /// <param name="printLetters">if true print each letter to console</param>
        /// <returns>the precentage of the correct results</returns>
        public double Accuracy(List<InputDataStructure> testData , bool printEachLetterSummary = false, bool printLetters = false)
        {
            var numCorrect = 0;
            
            // for each vector in the test set calculate the output of the network and check his label
            foreach (var dataStruct in testData)
            {
                // extract the input vector
                double[] xValues = dataStruct.GetDataVectorCopy();

                // extract the target vector
                double[] tValues = dataStruct.GetTargetVectorCopy();

                // compute the output values of the network for the current test vector
                var yValues = FeedForward(xValues); 

                // get the index of largest value - and choose this label
                var maxIndex = MaxIndex(yValues);
                var readlMaxIndex = MaxIndex(tValues);

                if (printLetters) Console.WriteLine(dataStruct);
                ;
                if (printEachLetterSummary)
                    Console.WriteLine("Computed NN value is :" + maxIndex +  ", Real target value is : " +  readlMaxIndex);

                // if the network retuned a corrrect output increase the counter
                if (tValues[maxIndex].Equals(1.0)) ++numCorrect;
            }
            
            // return the network accuracy on the specified data
            return testData.Count != 0 ? ((double)numCorrect) / testData.Count : 0;
        }

        #region Activation functions helpers

        /// <summary>
        /// Calculate and return the sigmoid function value of a double
        /// </summary>
        /// <param name="x">input value</param>
        /// <returns>sigmoid return value of x</returns>
        private static double SigmoidFunction(double x)
        {
            return 1 / (1 + Math.Exp(-1 * x));
        }

        /// <summary>
        /// Calculte and return the derviative of the sigmoid function
        /// </summary>
        /// <param name="x">input value</param>
        /// <param name="sigmoidOutput">true if the input value if already the output of the sigmoid function</param>
        /// <returns>sigmoid derviative return falue of x</returns>
        private static double SigmoidFunctionDerviative(double x , bool sigmoidOutput = false)
        {
            // if the input value is already the output of the sigmoid function
            if (sigmoidOutput)
                return (1 - x)*x;
            
            // else calculate the sigmoid value
            return (1 - SigmoidFunction(x))*SigmoidFunction(x);
        }

        /// <summary>
        /// Calculate and return the hyperTan function value of a double
        /// </summary>
        /// <param name="x">input value</param>
        /// <returns>hyperTan return value of x</returns>
        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            if (x > 20.0) return 1.0;
            return Math.Tanh(x);
        }

        /// <summary>
        /// Calculte and return the derviative of the hypertan function
        /// </summary>
        /// <param name="x">input value</param>
        /// <param name="hyperTanOutput">true if the input value if already the output of the hyperTan function</param>
        /// <returns>hyperTan derviative return falue of x</returns>
        private static double HyperTanFunctionDerviative(double x, bool hyperTanOutput = false)
        {
            if (hyperTanOutput)
                // derivative of tanh = (1 - y) * (1 + y)
                return (1 - x) * (1 + x);

            var hyperTanVal = HyperTanFunction(x);
            return (1 - hyperTanVal)*(1 + hyperTanVal);
        }

        /// <summary>
        /// Calculate and return the softmax values for a vector (better performance on a vector)
        /// </summary>
        /// <param name="sums">vector of linear summing</param>
        /// <returns></returns>
        private static double[] SoftmaxFunction(double[] sums)
        {
            // does all output nodes at once so scale doesn't have to be re-computed each time
            // 1. determine max output sum
            var max = sums[0];
            for (var i = 0; i < sums.Length; ++i)
                if (sums[i] > max) max = sums[i];

            // 2. determine scaling factor -- sum of exp(each val - max)
            var scale = 0.0;
            for (var i = 0; i < sums.Length; ++i)
                scale += Math.Exp(sums[i] - max);

            var result = new double[sums.Length];
            for (var i = 0; i < sums.Length; ++i)
                result[i] = Math.Exp(sums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }

        /// <summary>
        /// Calculte and return the derviative of the softmax function
        /// </summary>
        /// <param name="x">input value</param>
        /// <param name="softmaxOutput">true if the input value if already the output of the softmax function</param>
        /// <returns>softmax derviative return falue of x</returns>
        private static double SoftmaxFunctionDerviative(double x, bool softmaxOutput = false)
        {
            if (softmaxOutput)
                // derviative of soft max is the same as log-sigmoid
                return SigmoidFunctionDerviative(x, true);

            // need to implement for the cases where the input value is not already a softmax output
            throw new NotImplementedException();
        }


        #endregion

        #region Helpers

        /// <summary>
        /// Create a matrix of doubles 
        /// </summary>
        /// <param name="rows">number of rows</param>
        /// <param name="cols">number of colums</param>
        /// <returns>the newly created matrix</returns>
        private static double[][] MakeMatrix(int rows, int cols)
        {
            var result = new double[rows][];
            for (var r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }

        /// <summary>
        /// return the index of the max value in the vector
        /// </summary>
        /// <param name="vector">input vector</param>
        /// <returns>max value index</returns>
        private static int MaxIndex(double[] vector)
        {
            if (vector == null || vector.Length == 0) throw new ArgumentNullException("vector");

            // initialize return value
            var largestIndex = 0;
            var largestVal = vector[0];

            for (var i = 1; i < vector.Length; ++i)
            {
                // if we found a larger value
                if (vector[i] > largestVal)
                {
                    // update the largest index
                    largestVal = vector[i];
                    largestIndex = i;
                }
            }

            // return the largest index
            return largestIndex;
        }

        /// <summary>
        /// Shuffle the input index array
        /// </summary>
        /// <param name="indexArr">the index arr to shuffle</param>
        private static void Shuffle(int[] indexArr)
        {
            for (var i = 0; i < indexArr.Length; ++i)
            {
                var r = _rnd.Next(i, indexArr.Length);
                var tmp = indexArr[r];
                indexArr[r] = indexArr[i];
                indexArr[i] = tmp;
            }
        }

        #endregion

    }
}