/****************************************
 * Neural Networks - Project No1 
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

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

        private readonly double[][] _inputToHiddenWeights; 
        private readonly double[] _hiddenBiases;
        private readonly double[] _hiddenOutputs;

        private readonly double[][] _hiddenToOutputWeights;
        private readonly double[] _outputBiases;

        private readonly double[] _outputs;

        // momentum vectors
        private readonly double[][] _inputToHiddenPrevWeightsDelta;  
        private readonly double[] _hiddenPrevBiasesDelta;
        private readonly double[][] _hiddenToOutputPrevWeightsDelta;
        private readonly double[] _outputPrevBiasesDelta; 

        #endregion

        /// <summary>
        /// Cto'r
        /// </summary>
        /// <param name="numInput">number of input neurons</param>
        /// <param name="numHidden">number of hidden layer neurons</param>
        /// <param name="numOutput">number of output layer neurons, used also for labeling</param>
        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            // initialize new random variable for initializing weights
            _rnd = new Random(0); 
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;

            // initialize weights arrays
            _inputs = new double[numInput];

            _inputToHiddenWeights = CreateMatrix(numInput, numHidden);
            _hiddenBiases = new double[numHidden];
            _hiddenOutputs = new double[numHidden];

            _hiddenToOutputWeights = CreateMatrix(numHidden, numOutput);
            _outputBiases = new double[numOutput];

            _outputs = new double[numOutput];

            _inputToHiddenPrevWeightsDelta = CreateMatrix(numInput, numHidden);
            _hiddenPrevBiasesDelta = new double[numHidden];
            _hiddenToOutputPrevWeightsDelta = CreateMatrix(numHidden, numOutput);
            _outputPrevBiasesDelta = new double[numOutput];

            // initialize the weights randomaly
            InitializeWeights();
        } 

        /// <summary>
        /// initialize the weights and biases to small random values between -0.01 to 0.01
        /// </summary>
        private void InitializeWeights()
        {
            const double lo = -0.01;
            const double hi = 0.01;

            for (var i = 0; i < _numInput; ++i)
                for (var j = 0; j < _numHidden; ++j)
                    _inputToHiddenWeights[i][j] = (hi - lo) * _rnd.NextDouble() + lo;
            for (var i = 0; i < _numHidden; ++i)
                _hiddenBiases[i] = (hi - lo) * _rnd.NextDouble() + lo;
            for (var i = 0; i < _numHidden; ++i)
                for (var j = 0; j < _numOutput; ++j)
                    _hiddenToOutputWeights[i][j] = (hi - lo) * _rnd.NextDouble() + lo;
            for (var i = 0; i < _numOutput; ++i)
                _outputBiases[i] = (hi - lo) * _rnd.NextDouble() + lo;
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

            // hidden nodes sums 
            var hiddenBeforeActivation = new double[_numHidden]; 

            // output nodes sums
            var outputBeforeActivation = new double[_numOutput]; 

            // copy input values to the inner input vector
            Array.Copy(inputValues, _inputs, inputValues.Length);

            // calculate hidden neurons values
            for (var hiddenIdx = 0; hiddenIdx < _numHidden; ++hiddenIdx)
            {
                // calculate the sum of weights * inputs for the hidden neuron
                for (var inputIdx = 0; inputIdx < _numInput; ++inputIdx)
                    hiddenBeforeActivation[hiddenIdx] += _inputs[inputIdx] * _inputToHiddenWeights[inputIdx][hiddenIdx];

                // add the bias to the neuron
                hiddenBeforeActivation[hiddenIdx] += _hiddenBiases[hiddenIdx];
            }

            // apply the activation function to the hidden layer
            for (var i = 0; i < _numHidden; ++i) 
                _hiddenOutputs[i] = SigmoidFunction(hiddenBeforeActivation[i]);

            // calculate outputneurons values
            for (var outputIdx = 0; outputIdx < _numOutput; ++outputIdx)
            {
                // calculate the sum of weights * hiddens for the output neurons
                for (var hiddenIdx = 0; hiddenIdx < _numHidden; ++hiddenIdx)
                    outputBeforeActivation[outputIdx] += _hiddenOutputs[hiddenIdx] * _hiddenToOutputWeights[hiddenIdx][outputIdx];

                // add biases to input-to-hidden sums
                outputBeforeActivation[outputIdx] += _outputBiases[outputIdx];
            }
                
            // apply the activation function to the output layer
            for (var i = 0; i < _numOutput; ++i)
                _outputs[i] = SigmoidFunction(outputBeforeActivation[i]);

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

            var outputGradients = new double[_numOutput];
            var hiddenGradients = new double[_numHidden]; 

            // calculate output gradients
            for (var outputGradIdx = 0; outputGradIdx < outputGradients.Length; ++outputGradIdx)
                outputGradients[outputGradIdx] = SigmoidFunctionDerviative(_outputs[outputGradIdx], true)
                    * (tValues[outputGradIdx] - _outputs[outputGradIdx]);

            // update hidden to output weights
            for (var hiddenIdx = 0; hiddenIdx < _hiddenToOutputWeights.Length; ++hiddenIdx)
            {
                for (var outputIdx = 0; outputIdx < _hiddenToOutputWeights[0].Length; ++outputIdx)
                {
                    // calculate the new delta and update the weights
                    var delta = learnRate * outputGradients[outputIdx] * _hiddenOutputs[hiddenIdx];
                    _hiddenToOutputWeights[hiddenIdx][outputIdx] += delta;

                    // add momentum using previous delta
                    _hiddenToOutputWeights[hiddenIdx][outputIdx] += momentum * _hiddenToOutputPrevWeightsDelta[hiddenIdx][outputIdx];

                    // save the delta for momentum
                    _hiddenToOutputPrevWeightsDelta[hiddenIdx][outputIdx] = delta;
                }
            }

            // update output biases
            for (var outputIdx = 0; outputIdx < _outputBiases.Length; ++outputIdx)
            {
                // calculate the new dalta and update the bias
                var delta = learnRate * outputGradients[outputIdx];
                _outputBiases[outputIdx] += delta;

                // add momentum using previous delta
                _outputBiases[outputIdx] += momentum * _outputPrevBiasesDelta[outputIdx];

                // save the delta for momentum
                _outputPrevBiasesDelta[outputIdx] = delta;
            }

            // calculate hidden gradients
            for (var hiddenGradIdx = 0; hiddenGradIdx < hiddenGradients.Length; ++hiddenGradIdx)
            {
                var sum = 0.0;

                for (var j = 0; j < _numOutput; ++j) 
                    sum += outputGradients[j] * _hiddenToOutputWeights[hiddenGradIdx][j];
                
                hiddenGradients[hiddenGradIdx] = SigmoidFunctionDerviative(_hiddenOutputs[hiddenGradIdx], true) * sum;
            }

            // update input to hidden weights
            for (var inputIdx = 0; inputIdx < _inputToHiddenWeights.Length; ++inputIdx)
            {
                for (var hiddenIdx = 0; hiddenIdx < _inputToHiddenWeights[0].Length; ++hiddenIdx)
                {
                    // calculate the new delta and update the weights
                    var delta = learnRate * hiddenGradients[hiddenIdx] * _inputs[inputIdx]; 
                    _inputToHiddenWeights[inputIdx][hiddenIdx] += delta;

                    // add momentum using previous delta
                    _inputToHiddenWeights[inputIdx][hiddenIdx] += momentum * _inputToHiddenPrevWeightsDelta[inputIdx][hiddenIdx];

                    // save the delta for momentum 
                    _inputToHiddenPrevWeightsDelta[inputIdx][hiddenIdx] = delta; 
                }
            }

            // update hidden biases
            for (var hiddenIdx = 0; hiddenIdx < _hiddenBiases.Length; ++hiddenIdx)
            {
                // calculate the new dalta and update the bias
                var delta = learnRate * hiddenGradients[hiddenIdx];
                _hiddenBiases[hiddenIdx] += delta;

                // add momentum using previous delta
                _hiddenBiases[hiddenIdx] += momentum * _hiddenPrevBiasesDelta[hiddenIdx]; 

                // save the delta for momentum
                _hiddenPrevBiasesDelta[hiddenIdx] = delta;
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
            
            // create indexed array
            var sequence = HebLettersBackPropProgram.CreateIndexedArray(trainData.Count, shuffel:false);

            Console.Write("Starting epoch number : ");

            // while we didn't get to the epoch limit
            while (epoch < maxEpochs)
            {
                if (epoch % 100 == 0) Console.Write(epoch+ " ");

                // calculate meanSquaredError on the training set to see that we are not over fitting
                var mse = MeanSquaredError(trainData);

                // if we got to the desired mse, stop training
                if (mse < mseLimit) break;

                // run over the training data in random order
                HebLettersBackPropProgram.Shuffle(sequence); 

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
            Console.WriteLine();
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
        public double Accuracy(List<InputDataStructure> testData , bool printEachLetterSummary = false)
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

                if (printEachLetterSummary)
                {
                    Console.WriteLine(dataStruct);
                    Console.WriteLine("Computed NN value is :" + maxIndex +  ", Real target value is : " +  readlMaxIndex);
                }

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

        #endregion

        #region Helpers

        /// <summary>
        /// Create a matrix of doubles 
        /// </summary>
        /// <param name="rows">number of rows</param>
        /// <param name="cols">number of colums</param>
        /// <returns>the newly created matrix</returns>
        private static double[][] CreateMatrix(int rows, int cols)
        {
            var result = new double[rows][];
            for (var rowIndex = 0; rowIndex < result.Length; ++rowIndex)
                result[rowIndex] = new double[cols];
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

        #endregion
    }
}