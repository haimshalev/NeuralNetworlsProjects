using System;
using System.Collections.Generic;

namespace SelfOrgenizedMapNamespace
{
    /// <summary>
    /// Wrapper for a neighorhood dictionary
    /// </summary>
    public abstract class Topology
    {
        /// <summary>
        /// for each neuron (key is the neuron idx) saves the list of neighboors
        /// </summary>
        protected Dictionary<int, List<int>> _topolgyDictionary = new Dictionary<int, List<int>>();

        protected List<KeyValuePair<int, int>> _neighboorhoodPairs;

        /// <summary>
        /// used for initializing the topology
        /// </summary>
        /// <param name="numOfNeurons">number of neurons used in the kohonen layer</param>
        abstract public void InitializeTopology(int numOfNeurons);

        /// <summary>
        /// Returns the list of neighboors of the neuron 
        /// 
        /// For now only first degree neighberhood
        /// </summary>
        /// <param name="neuronIdx">the current neuron</param>
        /// <returns></returns>
        public List<int> GetNeighbourhood(int neuronIdx/*, int distance*/)
        {
            if (_topolgyDictionary.ContainsKey(neuronIdx))
                return _topolgyDictionary[neuronIdx];
            throw new ArgumentOutOfRangeException();
        }

        /// <returns>neighborhood list</returns>
        public List<KeyValuePair<int, int>> GetNeigborhoodPairs()
        {
            if (_neighboorhoodPairs != null) return _neighboorhoodPairs;

            // create the list for at the first time
            _neighboorhoodPairs = new List<KeyValuePair<int, int>>();
            foreach (var key in _topolgyDictionary.Keys)
                foreach (var val in _topolgyDictionary[key])
                    if (!_neighboorhoodPairs.Contains(new KeyValuePair<int, int>(val, key)))
                        _neighboorhoodPairs.Add(new KeyValuePair<int, int>(key, val));

            return _neighboorhoodPairs;
        }
    }

    /// <summary>
    /// Creates a line of neurons topology
    /// </summary>
    public class LineTopology : Topology
    {
        public override void InitializeTopology(int numOfNeurons)
        {
            for (int i = 0; i < numOfNeurons - 1; i++)
                _topolgyDictionary[i] = new List<int> { i + 1 };

            _topolgyDictionary[numOfNeurons - 1] = new List<int>();

            for (int i = 1; i < numOfNeurons; i++)
                _topolgyDictionary[i].Add(i - 1);
        }
    }

    /// <summary>
    /// Creates a circle of neurons topology
    /// </summary>
    public class CircleTopology : LineTopology
    {
        public override void InitializeTopology(int numOfNeurons)
        {
            // create a line topology
            base.InitializeTopology(numOfNeurons);

            // close both ends together
            _topolgyDictionary[0].Add(numOfNeurons - 1);
            _topolgyDictionary[numOfNeurons - 1].Add(0);
        }
    }
}
