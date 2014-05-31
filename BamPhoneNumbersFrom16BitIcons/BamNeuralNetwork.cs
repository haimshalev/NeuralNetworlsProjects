/****************************************
 * Neural Networks - Project No2 
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;
using System.Linq;

namespace BamPhoneNumbersFrom16BitIcons
{
    class BamNeuralNetwork
    {
        private readonly int _numberOfInputNeurons;
        private readonly int _numberOfOutputNeurons;
        private readonly int[][] _transformationMatrix;
        private readonly int[][] _transformationMatrixReverse;

        private const int BamThreshold = 0;

        /// <summary>
        /// Cto'r
        /// </summary>
        /// <param name="numberOfInputNeurons">the size of the input vectors</param>
        /// <param name="numberOfOutputNeurons">the size of the output vectors</param>
        public BamNeuralNetwork(int numberOfInputNeurons, int numberOfOutputNeurons)
        {
            // set internal properties
            _numberOfInputNeurons = numberOfInputNeurons;
            _numberOfOutputNeurons = numberOfOutputNeurons;

            // initializing the transformation matrices
            _transformationMatrix = new int[_numberOfInputNeurons][];
            for (var i = 0; i < _numberOfInputNeurons; i++)
                _transformationMatrix[i] = new int[_numberOfOutputNeurons];

            _transformationMatrixReverse = new int[_numberOfOutputNeurons][];
            for (var i = 0; i < _numberOfOutputNeurons; i++)
                _transformationMatrixReverse[i] = new int[_numberOfInputNeurons];
        }

        /// <summary>
        /// Add an association to the neural network and updates the inner transformation matrix
        /// </summary>
        /// <param name="input">input vector</param>
        /// <param name="output">output vector</param>
        public void AddAssociation(int[] input, int[] output)
        {
            if (input.Length != _numberOfInputNeurons || output.Length != _numberOfOutputNeurons)
                throw new ArgumentException("AddAssociation filed: one of the input or output vectors is from incorrect size");

            // update the values of the transformation matrix
            for (var i = 0; i < _numberOfInputNeurons; i++)
                for (var j = 0; j < _numberOfOutputNeurons; j++)
                {
                    _transformationMatrix[i][j] += input[i] * output[j];
                    _transformationMatrixReverse[j][i] += input[i] * output[j];
                }
        }

        /// <summary>
        /// Clear the inner transformation matrix
        /// </summary>
        public void RemoveAssociations()
        {
            for (var i = 0; i < _numberOfInputNeurons; i++)
                for (var j = 0; j < _numberOfOutputNeurons; j++)
                {
                    _transformationMatrix[i][j] = 0;
                    _transformationMatrixReverse[j][i] = 0;
                }
        }

        /// <summary>
        /// Associate the input vector to one of the stored associations
        /// </summary>
        /// <param name="input">the input vector to associate</param>
        /// <param name="output">optional parameter, the desired output vector</param>
        public void Associate(int[] input, int[] output)
        {
            var isForwardStable = false;
            var isBackwardStable = false;

            // while the 2 vectors are unstable (not changing from iteration to iteration) 
            while (!isBackwardStable && !isForwardStable)
            {
                // propagate forward and propagate backward
                isForwardStable = PropagateLayer(_transformationMatrix, input, output);
                isBackwardStable = PropagateLayer(_transformationMatrixReverse, output, input);
            }

            PropagateLayer(_transformationMatrix, input, output);
            PropagateLayer(_transformationMatrixReverse, output, input);
        }

        /// <summary>
        /// propagate the input vector to the output layer using the given transformation matrix
        /// </summary>
        /// <param name="transformationMatrix">the transformation matrix to the other layer</param>
        /// <param name="input">the input vector</param>
        /// <param name="output">the current association</param>
        /// <returns>true if the output vector stays the same</returns>
        private bool PropagateLayer(int[][] transformationMatrix, int[] input, int[] output)
        {
            // initialize the network to stable
            var stable = true;

            // run over all the output values
            for (var i = 0; i < output.Length; i++)
            {
                // vector multiplication with the correct matrix
                int sum;
                if (input.Length != transformationMatrix.Length)
                    sum = input.Select((t, j) => transformationMatrix[i][j]*t).Sum();
                else
                    sum = input.Select((t, j) => transformationMatrix[j][i]*t).Sum();
                
                // threshold the output
                var outputVal = Threshold(sum);

                // if we didn't get the desired output, unstable
                if (outputVal != output[i])
                {
                    stable = false;
                    output[i] = outputVal;
                }
            }

            return stable;
        }

        /// <summary>
        /// return 1 if the input value greater then the threshold, else -1
        /// </summary>
        /// <param name="x">input value</param>
        /// <param name="threshold">checked threshold</param>
        /// <returns>1 if the input value greater then the threshold, else -1</returns>
        private static int Threshold(double x, int threshold = BamThreshold)
        {
            if (x > threshold) return 1;
            return -1;
        }
    }
}
