using System;

namespace BamPhoneNumbersFrom16BitIcons
{
    class BamPhoneNumbersFrom16BitIconsProgram
    {
        public const int NumberTrainingPoints = 5;

        static void Main()
        {
            #region Prepare the data

            // initializing input vectors - 16 bit icons
            var inputVectors = new int[NumberTrainingPoints][];
            inputVectors[0] = new[] { 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1 };
            inputVectors[1] = new[] { 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0 };
            inputVectors[2] = new[] { 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1 };
            inputVectors[3] = new[] { 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0 };
            inputVectors[4] = new[] { 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0 };


            // initialize output vectors - phone numbers
            var outputVectors = new[] { "0523331122", "0526642720", "0538795012", "0548960551", "0508656789"};

            #endregion

            // initialize the BAM neural network wrapper
            var bamNeuralNetworkWrapper = new BamNetworkPhoneToIconsWrapper(inputVectors, outputVectors);
            
            #region Retrival

            // TODO: setting new test vectors
            
            // TODO: Associate the new test vectors
            
            // TODO: output results

            #endregion
        }
    }
}


