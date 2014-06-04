/****************************************
 * Neural Networks - Project No3
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;
using System.Collections.Generic;
using System.Linq;

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

        /// <summary>
        /// list of edges in the topology
        /// </summary>
        protected List<KeyValuePair<int, int>> _neighboorhoodPairs;

        /// <summary>
        /// matrix which contains the distance between each neuron to the other
        /// </summary>
        protected int[][] _distanceMatrix;

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

        /// <returns>the distance between two neurons on the specified topology</returns>
        public int GetDistance(int neuron1Idx, int neuron2Idx)
        {
            if (_distanceMatrix != null) return _distanceMatrix[neuron1Idx][neuron2Idx];

            // if the distance matrix was not created , create it
            _distanceMatrix = new int[_topolgyDictionary.Count][];
                
            // for every node create it's distance vector to every one else using dijkstra algorithm
            for (var i = 0; i < _distanceMatrix.Length; i++)
            {
                // create an array with -1 on every cell
                _distanceMatrix[i] = Enumerable.Repeat(-1,_distanceMatrix.Length).ToArray();

                var nodes = Enumerable.Range(0, _topolgyDictionary.Count).ToList();
                var thisRoundNodes = new List<int>();
                var nextRoundNodes = new List<int>{i};
                var distance = 0;
                    
                while (nodes.Count() != 0 && nextRoundNodes.Count() != 0)
                {
                    thisRoundNodes.AddRange(nextRoundNodes);
                    nextRoundNodes.Clear();

                    foreach (var thisRoundNode in thisRoundNodes)
                    {
                        // remove the node from the discovery list
                        nodes.Remove(thisRoundNode);

                        // set the distance
                        if (_distanceMatrix[i][thisRoundNode] == -1) _distanceMatrix[i][thisRoundNode]  = distance;

                        // add his neighboors to the next round list
                        nextRoundNodes.AddRange(GetNeighbourhood(thisRoundNode).Intersect(nodes));
                    }

                    thisRoundNodes.Clear();
                    distance++;
                }
            }

            return _distanceMatrix[neuron1Idx][neuron2Idx];
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
