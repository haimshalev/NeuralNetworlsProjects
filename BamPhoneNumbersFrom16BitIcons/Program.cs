using System;

namespace BamPhoneNumbersFrom16BitIcons
{
    class Program
    {
        public const int NumberInputNeurons = 15;
        public const int NumberOutputNeurons = 3;
        public const int NumberTrainingPoints = 2;

        static void Main()
        {
            #region Prepare the data

            // initializing input vectors
            var inputVectors = new int[NumberTrainingPoints][];
            inputVectors[0] = new[] { 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1 };
            inputVectors[1] = new[] { -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1 };

            // initialize output vectors 
            var outputVectors = new int[NumberTrainingPoints][];
            outputVectors[0] = new[] { 1, 1, -1 };
            outputVectors[1] = new[] { 1, 1, 1 }; 

            // TODO: need to convert the data to biPolar

            #endregion

            // initialize the BAM neural network
            var bamNeuralNetwork = new BamNeuralNetwork(NumberInputNeurons, NumberOutputNeurons);
            
            // add the associations to the network
            for (var i = 0; i < NumberTrainingPoints; i++)
                bamNeuralNetwork.AddAssociation(inputVectors[i], outputVectors[i]);

            #region Retrival

            // setting new inputs
            var testVector = new[] { 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1 };

            // TODO: create random biPolar output vector
            var outputVector = new[] {-1, -1, 1};

            // Associate the test vector
            bamNeuralNetwork.Associate(testVector , outputVector);

            // output the result vector
            Console.WriteLine("output vector : " + String.Join(",",outputVector));

            #endregion
        }
    }
}


