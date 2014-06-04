/****************************************
 * Neural Networks - Project No3
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;
using System.Linq;

namespace SelfOrgenizedMapNamespace
{
    /// <summary>
    /// base class for the selfOrgenizedMap
    /// </summary>
    public abstract class SelfOrgenizedMap
    {
        #region Properties

        /// <summary>
        /// The weights of the output set to the coordinates of each neuron
        /// neurons.
        /// </summary>
        protected double[][] _weights;

        /// <summary>
        /// the biases for each neuron - used for not selecting the same neuron all the time
        /// </summary>
        protected double[] _biases;

        /// <summary>
        /// the dimension of each datapoint
        /// </summary>
        protected int _inputDimension;

        /// <summary>
        /// the number of Kohonen neurons (number of clasters)
        /// </summary>
        protected int _outputNeuronsCount;

        /// <summary>
        /// UI window - used for updating the GUI
        /// </summary>
        protected MainWindow _mainWindow;

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
        /// Start learning the input data and update the main window 
        /// </summary>
        public abstract void Learn();
    }

    /// <summary>
    /// A topology generic self organizing map neural network
    /// </summary>
    public class SelfOrgnizedMap<T> : SelfOrgenizedMap
        where T : Topology , new()
    {
        /// <summary>
        /// the Kohonen layer topology
        /// </summary>
        private readonly Topology _topology;

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
            
            _topology = new T();
            _topology.InitializeTopology(numberOfClusters);

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
        public override void Learn()
        {
            var rand = new Random(DateTime.Now.Millisecond);
            var iteration = 0;
            const double startLearnRate = 0.5;
            const double endLearnRate = 0.1;
            const double startAttraction = 3.0e0;
            const double endAttraction = 1.0e-1;

            double learnRate = startLearnRate + ((double)iteration / (NumEpoches * Data.Length)) * (endLearnRate - startLearnRate);

            // start learning - on every epoch we go over all the points in the train data
            for (int i = 0; i < NumEpoches; i++)
            {
                // run over all the input data randomly
                var alldata = Data.ToList();
                while (alldata.Count != 0)
                {
                    // decrease the learn rate
                    learnRate = startLearnRate + ((double)iteration / (NumEpoches * Data.Length)) * (endLearnRate - startLearnRate);

                    // decrease the attraction 
                    double attraction = startAttraction + ((double) iteration/(NumEpoches*Data.Length))*(endAttraction - startAttraction);

                    // update the gui if necessery
                    if (iteration % MainWindowRefreshRate == 0)
                        _mainWindow.UpdateWindow(_weights, _topology.GetNeigborhoodPairs(), learnRate, i, iteration);

                    var input = alldata.ElementAt(rand.Next(alldata.Count));
                    alldata.Remove(input);

                    // Find the closest cluster
                    var closestNeuronIdx = FindClosestNeuron(input);

                    // update biases
                    UpdateBiases(closestNeuronIdx);

                    // change the neuron location by updating it's weights to be closer to the chosen input
                    for (int neuronToUpdateIdx = 0; neuronToUpdateIdx < _weights.Length; neuronToUpdateIdx++)
                        ModifyWights(neuronToUpdateIdx, input, closestNeuronIdx, learnRate, attraction);
                    

                    iteration ++;
                }
            }

            _mainWindow.UpdateWindow(_weights, _topology.GetNeigborhoodPairs(), learnRate, NumEpoches, iteration);
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
            const double biasChangeRate = 20;

            for (int i = 0; i < _biases.Length; i++)
            {
                if (i != chosenNeuronIdx && _biases[i] != 0) _biases[i] -= biasChangeRate/4;
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
        private void ModifyWights(int neuronToChangeIdx, double[] dataInput, int closestNeuronIdx, double alpha, double attraction)
        {
            for (int i = 0; i < dataInput.Length; i++)
                _weights[neuronToChangeIdx][i] += alpha*NeighberhoodFunction(neuronToChangeIdx, closestNeuronIdx, attraction)*
                                                  (dataInput[i] - _weights[neuronToChangeIdx][i]);
        }

        /// <summary>
        /// returns the amount of change according to the topology
        /// </summary>
        /// <param name="neuronToChange">the neuron which his weights are modified</param>
        /// <param name="winnerNeuron">the neuron which represent the winner claster</param>
        /// <param name="attraction">attraction factor</param>
        /// <returns>amount of modification acording to the topology</returns>
        private double NeighberhoodFunction(int neuronToChange, int winnerNeuron, double attraction)
        {
            var distanceBetweenNeurons = _topology.GetDistance(neuronToChange, winnerNeuron);

            //using maxican hat neighberhood function
            return Math.Exp(
                     ((distanceBetweenNeurons*distanceBetweenNeurons) / (2*attraction*attraction)) *
                     (1 - (2*distanceBetweenNeurons*distanceBetweenNeurons) / (2*attraction*attraction))
                     );
        }
    }
}

