using System;
using System.Linq;

namespace SelfOrgenizedMap
{
    /// <summary>
    /// A self organizing map neural network.
    /// </summary>
    [Serializable]
    public class SelfOrgnizedMap 
    {
        #region Properties

        /// <summary>
        /// The weights of the output set to the coordinates of each neuron
        /// neurons.
        /// </summary>
        private readonly double[][] _weights;

        /// <summary>
        /// the biases for each neuron - used for not selecting the same neuron all the time
        /// </summary>
        private readonly double[] _biases;

        /// <summary>
        /// the dimension of each datapoint
        /// </summary>
        private readonly int _inputDimension;

        /// <summary>
        /// the number of Kohonen neurons (number of clasters)
        /// </summary>
        private readonly int _outputNeuronsCount;

        /// <summary>
        /// UI window - used for updating the GUI
        /// </summary>
        private readonly MainWindow _mainWindow;

        /// <summary>
        /// data to claster
        /// </summary>
        public double[][] Data;

        /// <summary>
        /// num of rounds to learn
        /// </summary>
        public int NumEpoches;

        /// <summary>
        /// main window canvas refresh rate
        /// </summary>
        public int MainWindowRefreshRate; 

        #endregion

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="inputDimension">Number of input neurons</param>
        /// <param name="numberOfClusters">Number of output neurons</param>
        /// <param name="window">the gui window to notify</param>
        public SelfOrgnizedMap(int inputDimension, int numberOfClusters, MainWindow window)
        {
            _inputDimension = inputDimension;
            _outputNeuronsCount = numberOfClusters;

            _mainWindow = window;

            // initialize the weight matrix - and set all neurons at the center of the window
            _weights = new double[numberOfClusters][];
            for (var i = 0; i < numberOfClusters; i++)
                _weights[i] = new double[]{50, 50};   
            
            // initialize the biases for each neuron
            _biases = new double[numberOfClusters];
        }

        /// <summary>
        /// Start learning the input data and update the main window 
        /// </summary>
        public void Learn()
        {
            var rand = new Random(DateTime.Now.Millisecond);
            double learnRate = 1;

            // start learning - on every epoch we go over all the points in the train data
            for (int i = 0; i < NumEpoches; i++)
            {
                // update the gui if necessery
                if (i % MainWindowRefreshRate== 0)
                    _mainWindow.UpdateWindow(_weights, learnRate, i);

                // decrease the learn rate
                learnRate = 1 - ((double)i/NumEpoches);
                
                // run over all the input data randomly
                var alldata = Data.ToList();
                while (alldata.Count != 0)
                {
                    var input = alldata.ElementAt(rand.Next(alldata.Count));
                    alldata.Remove(input);

                    // Find the closest cluster
                    var neuronIdx = FindClosestNeuron(input);

                    // update biases
                    UpdateBiases(neuronIdx);

                    // change the neuron location by updating it's weights to be closer to the chosen input
                    for (var cluster = 0; cluster < _outputNeuronsCount; cluster++)
                        for (int j = 0; j < _inputDimension; j++)
                            ModifyWights(cluster, input, neuronIdx, learnRate);
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
                var dist = EuclideanDistance(input, _weights[i]) + _biases[i];
                if (dist < minDist)
                {
                    minDist = dist;
                    result = i;
                }
            }

            return result;
        }

        /// <summary>
        /// update the biases of the cluster neurons
        /// Increasing the bias of the chosen neuron
        /// Decreasing the bias of the other neurons
        /// </summary>
        /// <param name="chosenNeuronIdx"></param>
        private void UpdateBiases(int chosenNeuronIdx)
        {
            const double biasChangeRate = 1;

            for (int i = 0; i < _biases.Length; i++)
            {
                if (i != chosenNeuronIdx && _biases[i] != 0) _biases[i] -= biasChangeRate;
                else _biases[i] += biasChangeRate;
            }
        }

        /// <summary>
        /// Calculate the Euclidean distance between two vectors
        /// </summary>
        /// <param name="vec1">The first vector</param>
        /// <param name="vec2">The second vector</param>
        /// <returns>The distance.</returns>
        private static double EuclideanDistance(double[] vec1, double[] vec2)
        {
            return Math.Sqrt(vec1.Select((t, i) => t - vec2[i]).Sum(d => d * d));
        }

        /// <summary>
        /// Updates the weight of the neuron
        /// </summary>
        /// <param name="neuronToChangeIdx">the neuron which weights needs change</param>
        /// <param name="dataInput">data point which we want to cluster</param>
        /// <param name="closestNeuronIdx">chosen closest neuron idx </param>
        /// <param name="alpha">learn rate</param>
        private void ModifyWights(int neuronToChangeIdx, double[] dataInput, int closestNeuronIdx, double alpha)
        {
            for (int i = 0; i < dataInput.Length; i++)
                _weights[neuronToChangeIdx][i] += alpha*NeighberhoodFunction(neuronToChangeIdx, closestNeuronIdx)*
                                                  (dataInput[i] - _weights[neuronToChangeIdx][i]);
        }

        /// <summary>
        /// returns the amount of change according to the topology
        /// </summary>
        /// <param name="neuronToChange">the neuron which his weights are modified</param>
        /// <param name="winnerNeuron">the neuron which represent the winner claster</param>
        /// <returns>amoount of modification</returns>
        private double NeighberhoodFunction(int neuronToChange, int winnerNeuron)
        {
            if (neuronToChange == winnerNeuron)
                return 0.3;
            if (Math.Abs(neuronToChange - winnerNeuron) == 1)
                return 0.1;
            return 0;
        }
    }
}

