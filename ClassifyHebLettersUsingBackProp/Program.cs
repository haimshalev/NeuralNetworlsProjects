/****************************************
 * Neural Networks - Project No1 
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;
using System.Collections.Generic;
using System.IO;

namespace ClassifyHebLettersUsingBackProp
{
    internal class HebLettersBackPropProgram
    {
        #region Static Properties

        // input letters directory
        private const string LettersDirectoryPath = @"letters";

        // file types - for now using only png files
        private const string FileTypes = "*.PNG";

        // neural network structure
        const int NumInput = 100;
        const int NumHidden = 100;
        const int NumOutput = 5;

        // training properties
        const int MaxEpochs = 3000;
        const double LearnRate = 0.01;
        const double Momentum = 0.001;
        const double MseLimit = 0.05;

        #endregion

        private static void Main()
        {
            // create the vector db
            var allData = CreateVectorsDb(printLetters: false);

            Console.WriteLine("\nBegin neural network classification and prediction");
            Console.WriteLine("target-representation is aleph = 1 0 0 0 0, beith = 0 1 0 0 0, giemel = 0 0 1 0 0, daled = 0 0 0 1 0, hei = 0 0 0 0 1");
            
            // Run 5 folds cross validation
            CrossValidation(allData, 5);

            Console.WriteLine("\nEnd of program");
            Console.ReadLine();
        }

        /// <summary>
        /// Run cross validation on the data 
        /// </summary>
        /// <param name="allData">data set</param>
        /// <param name="numFolds">number of folds to perform</param>
        private static void CrossValidation(List<InputDataStructure> allData, int numFolds)
        {
            double avgTrainAccuracy = 0;
            double avgTestAccuracy = 0;

            for (var foldIdx = 0; foldIdx < numFolds; foldIdx++)
            {
                Console.WriteLine("\n\nStarting Fold #" + foldIdx);

                // Create the neural network structure
                Console.WriteLine("\nCreating a " + NumInput + "-input, " +
                                  NumHidden + "-hidden, " +
                                  NumOutput + "-output neural network");
                Console.WriteLine("Using logisitic sigmoid as activation functions on each layer");
                var nn = new NeuralNetwork(NumInput, NumHidden, NumOutput);

                // set the training properties and train the network on the training data
                Console.WriteLine("\nSetting maxEpochs = " + MaxEpochs +
                                  ", learnRate = " + LearnRate +
                                  ", momentum = " + Momentum +
                                  ", mseLimit = " + MseLimit + " for stopping condition");

                Console.WriteLine("Seperation input data to 80% training and 20% test");
                var trainData = new List<InputDataStructure>();
                var testData = new List<InputDataStructure>();
                //MakeTrainTest(allData, trainData, testData);
                MakeTrainTest(allData, trainData, testData, foldIdx);

                Console.WriteLine("Beginning training:\n");
                nn.Train(trainData, MaxEpochs, LearnRate, Momentum, MseLimit);
                Console.WriteLine("Training complete");

                // get the accuracy over the training data
                var trainAcc = nn.Accuracy(trainData, printEachLetterSummary: false);
                Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

                // get the accuracy over the test data
                var testAcc = nn.Accuracy(testData, printEachLetterSummary:true);
                Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));
                Console.ReadKey();

                avgTestAccuracy += testAcc;
                avgTrainAccuracy += trainAcc;
            }
            
            Console.WriteLine("\nAvgAccurracy on the training data =" + avgTrainAccuracy / numFolds);
            Console.WriteLine("\nAvgAccurracy on the test data =" + avgTestAccuracy / numFolds);
        }

        /// <summary>
        /// Seperate the data to training set and testing set (80% - 20%) 
        /// </summary>
        /// <param name="allData">input data set</param>
        /// <param name="trainData">output training set</param>
        /// <param name="testData">output testing set</param>
        /// <param name="foldIdx">cross validation fold index - used for choosing the data</param>
        private static void MakeTrainTest(List<InputDataStructure> allData, List<InputDataStructure> trainData, List<InputDataStructure> testData, int foldIdx)
        {
            for (var i = 0; i < allData.Count; i++)
            {
                if (i%5 == foldIdx)
                    testData.Add(allData[i].GetCopy());
                else trainData.Add(allData[i].GetCopy());
            }
        }

        /// <summary>
        /// Shuffle the input index array
        /// </summary>
        /// <param name="indexArr">the index arr to shuffle</param>
        public static void Shuffle(int[] indexArr)
        {
            var rnd = new Random(DateTime.Now.Millisecond);

            for (var i = 0; i < indexArr.Length; ++i)
            {
                var randomNextIndex = rnd.Next(i, indexArr.Length);
                var tmp = indexArr[randomNextIndex];
                indexArr[randomNextIndex] = indexArr[i];
                indexArr[i] = tmp;
            }
        }

        /// <summary>
        /// Create and return index array from specified size
        /// </summary>
        /// <param name="size">the size of the created array</param>
        /// <param name="shuffel">true uf the indeces should be in random order</param>
        /// <returns>indexed array</returns>
        public static int[] CreateIndexedArray(int size, bool shuffel = true)
        {
            var sequence = new int[size];
            for (var i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            if (shuffel) Shuffle(sequence);

            return sequence;
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