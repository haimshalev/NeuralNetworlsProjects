using System;
using System.Collections.Generic;
using System.IO;

namespace ClassifyHebLettersUsingBackProp
{
    internal class HebLettersBackPropProgram
    {
        #region Static Properties

        // input letters directory
        private const string LettersDirectoryPath = @"../../letters";

        // file types - for now using only png files
        private const string FileTypes = "*.PNG";

        // neural network structure
        const int NumInput = 100;
        const int NumHidden = 200;
        const int NumOutput = 5;

        // training properties
        const int MaxEpochs = 4000;
        const double LearnRate = 0.01;
        const double Momentum = 0.001;
        const double MseLimit = 0.020;

        #endregion

        private static void Main()
        {
            // create the vector db
            var allData = CreateVectorsDb(printLetters: true);

            Console.ReadLine();

            Console.WriteLine("\nBegin neural network classification and prediction");
            Console.WriteLine("X-data is x0, x1, x2, ... , x100");
            Console.WriteLine("Y-data is aleph = 1 0 0 0 0, beith = 0 1 0 0 0, giemel = 0 0 1 0 0, daled = 0 0 0 1 0, hei = 0 0 0 0 1");


            Console.WriteLine("Creating 80% training and 20% test data matrices");
            var trainData = new List<InputDataStructure>();
            var testData = new List<InputDataStructure>();
            MakeTrainTest(allData, trainData, testData);

            // Create the neural network structure
            Console.WriteLine("\nCreating a " + NumInput + "-input, " + 
                              NumHidden + "-hidden, " +
                              NumOutput + "-output neural network");
            Console.WriteLine("Using logisitic sigmoid as activation functions on each layer");
            var nn = new NeuralNetwork(NumInput, NumHidden, NumOutput);

            // set the training properties and train the network on the training data
            Console.WriteLine("Setting maxEpochs = " + MaxEpochs +
                              ", learnRate = " + LearnRate + 
                              ", momentum = " + Momentum + 
                              ", mseLimit = " + MseLimit + " for stopping condition");
            Console.WriteLine("\nBeginning training using incremental back-propagation\n");
            nn.Train(trainData, MaxEpochs, LearnRate, Momentum, MseLimit);
            Console.WriteLine("Training complete");

            // get the accuracy over the training data
            var trainAcc = nn.Accuracy(trainData, printEachLetterSummary: false, printLetters: false);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            // get the accuracy over the test data
            var testAcc = nn.Accuracy(testData, printEachLetterSummary: true, printLetters: true);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));

            Console.WriteLine("\nEnd of program\n");
            Console.ReadLine();
        }

        /// <summary>
        /// Seperate the data to training set and testing set (80% - 20%) 
        /// </summary>
        /// <param name="allData">input data set</param>
        /// <param name="trainData">output training set</param>
        /// <param name="testData">output testing set</param>
        private static void MakeTrainTest(List<InputDataStructure> allData, List<InputDataStructure> trainData, List<InputDataStructure> testData)
        {
            if (allData == null) throw new ArgumentNullException("allData");

            // split allData into 80% trainData and 20% testData
            var rnd = new Random(0);
            int totInput = allData.Count;

            // hard-coded 80-20 split
            var numOfTrainObjects = (int) (totInput*0.80); 

            // create a random sequence of indexes
            var sequence = new int[totInput]; 
            for (var i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (var i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            // the first objets go to training
            var si = 0;
            for (; si < numOfTrainObjects; ++si) 
                trainData.Add(allData[sequence[si]].GetCopy());
            
            // remainder to test data
            for (; si < totInput; ++si) 
                testData.Add(allData[sequence[si]].GetCopy());
        }

        /// <summary>
        /// Creates and returns vector db from the input letters located in the input folder
        /// </summary>
        /// <param name="printLetters">if true , prints each letter to the console</param>
        /// <returns>the vector db</returns>
        public static List<InputDataStructure> CreateVectorsDb(bool printLetters = false)
        {
            Console.WriteLine("Creating vector db from the letters in the input folder : " + LettersDirectoryPath);

            // get all the letters from the input directory
            string[] fileEntries = Directory.GetFiles(Path.GetFullPath(LettersDirectoryPath), FileTypes);

            // initialize the vectors db
            var allData = new List<InputDataStructure>();

            // for each letter, create it's represntation vector
            foreach (var fileEntry in fileEntries)
            {
                // create new heb letter input from the file 
                var letter = new HebLetterInputDataStructure(fileEntry);

                // add it to the input list
                allData.Add(letter);
                
                // print the current letter to the console
                if (printLetters) Console.WriteLine(letter);
            }

            // return the vector db
            return allData;
        }
   } 

} 