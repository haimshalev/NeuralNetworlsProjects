using System;
using System.Collections.Generic;
using System.Linq;

namespace SelfOrgenizedMap
{
    /// <summary>
    /// A self organizing map neural network.
    /// </summary>
    [Serializable]
    public class SelfOrgnizedMap 
    {
        /// <summary>
        /// The weights of the output set to the coordinates of each neuron
        /// neurons.
        /// </summary>
        private readonly double[][] _weights;

        private readonly int _inputNeuronsCount;
        private readonly int _outputNeuronsCount;

        private readonly MainWindow _mainWindow;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="inputDimension">Number of input neurons</param>
        /// <param name="numberOfClusters">Number of output neurons</param>
        /// <param name="window">the gui window to notify</param>
        public SelfOrgnizedMap(int inputDimension, int numberOfClusters, MainWindow window)
        {
            _inputNeuronsCount = inputDimension;
            _outputNeuronsCount = numberOfClusters;

            _mainWindow = window;

            // initialize the weight matrix - and set all neurons at the center of the window
            _weights = new double[numberOfClusters][];
            for (var i = 0; i < numberOfClusters; i++)
                _weights[i] = new double[]{50, 50};      
        }

        /// <summary>
        /// Start learning on the input data and update the main window 
        /// </summary>
        /// <param name="inputData">the data to claster</param>
        /// <param name="numOfEpoces">num of rounds to learn</param>
        /// <param name="refreshRate">main window canvas refresh rate
        ///                           TODO: add refresh points array 
        /// </param>
        public void Learn(double[][] inputData, int numOfEpoces, int refreshRate)
        {
            var rand = new Random(DateTime.Now.Millisecond);

            // start learning
            for (int i = 0; i < numOfEpoces; i++)
            {
                // update the gui
                if (numOfEpoces%refreshRate == 0)
                    _mainWindow.UpdateWindow(_weights);

                // run over all the input data randomly
                var alldata = inputData.ToList();
                while (alldata.Count != 0)
                {
                    var input = alldata.ElementAt(rand.Next(alldata.Count));
                    alldata.Remove(input);

                    // Find the cluster
                    var neuronIdx = FindClosestNeuron(input);

                    // change the neuron location by updating it's weights to be closer to the chosen input
                    for (var cluster = 0; cluster < _outputNeuronsCount; cluster++)
                        for (int j = 0; j < _inputNeuronsCount; j++)
                            ModifyWights(cluster, input, neuronIdx, i, function);
                }
            }
        }


        /// <summary>
        /// Classify the input into one of the output clusters
        /// </summary>
        /// <param name="input">The input datapoint</param>
        /// <returns>The cluster idx it was clasified to</returns>
        private int FindClosestNeuron(double[] input)
        {
            double minDist = Double.PositiveInfinity;
            int result = -1;

            for (var i = 0; i < _outputNeuronsCount; i++)
            {
                var dist = EuclideanDistance(input, _weights[i]);
                if (dist < minDist)
                {
                    minDist = dist;
                    result = i;
                }
            }

            return result;
        }

        /// <summary>
        /// Calculate the Euclidean distance between two vectors
        /// </summary>
        /// <param name="vec1">The first vector</param>
        /// <param name="vec2">The second vector</param>
        /// <returns>The distance.</returns>
        private static double EuclideanDistance(IEnumerable<double> vec1, IList<double> vec2)
        {
            return Math.Sqrt(vec1.Select((t, i) => t - vec2[i]).Sum(d => d * d));
        }

        /// <summary>
        /// Updates the weight of the neuron according to the other parameters
        /// </summary>
        /// <param name="neuronToChangeIdx">the neuron which weights needs change</param>
        /// <param name="dataInput">data point which we want to cluster</param>
        /// <param name="ClosestNeuronIdx">chosen closest neuron idx </param>
        /// <param name="epochNum">epoch num - used to decrease the bonding between the Kohonen neurons on later epoches</param>
        /// <param name="modificationFunction">the function which we use to modify the weights</param>
        private void ModifyWights(int neuronToChangeIdx, double[] dataInput, int ClosestNeuronIdx, int epochNum,
            ModificationFunction modificationFunction)
        {
            
        }
    }
}

