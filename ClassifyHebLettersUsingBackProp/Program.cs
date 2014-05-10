using System;
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

        // each file size
        private const int LetterHeight = 10;
        private const int LetterWidth = 10;
        private const int RepresentationVectorSize = LetterHeight*LetterWidth;

        // for now we are labeling only 5 letter - aleph, beith, giemel, daled, hei
        private const int NumOfLabels = 5;

        // the threshold for a specific pixel to be white
        private const int PixelThreshold = 200;

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
            var allData = CreateVectorsDb(printLetters: false);

            Console.WriteLine("\nBegin neural network classification and prediction");
            Console.WriteLine("X-data is x0, x1, x2, ... , x100");
            Console.WriteLine("Y-data is aleph = 1 0 0 0 0, beith = 0 1 0 0 0, giemel = 0 0 1 0 0, daled = 0 0 0 1 0, hei = 0 0 0 0 1");


            Console.WriteLine("Creating 80% training and 20% test data matrices");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, out trainData, out testData);

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
            var testAcc = nn.Accuracy(testData, printEachLetterSummary: false, printLetters: false);
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
        private static void MakeTrainTest(double[][] allData, out double[][] trainData, out double[][] testData)
        {
            // split allData into 80% trainData and 20% testData
            var rnd = new Random(0);
            int totRows = allData.Length;
            int numCols = allData[0].Length;

            var trainRows = (int) (totRows*0.80); // hard-coded 80-20 split
            var testRows = totRows - trainRows;

            trainData = new double[trainRows][];
            testData = new double[testRows][];

            var sequence = new int[totRows]; // create a random sequence of indexes
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            int si = 0; // index into sequence[]
            int j = 0; // index into trainData or testData

            for (; si < trainRows; ++si) // first rows to train data
            {
                trainData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(allData[idx], trainData[j], numCols);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < totRows; ++si) // remainder to test data
            {
                testData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(allData[idx], testData[j], numCols);
                ++j;
            }
        }

        /// <summary>
        /// print a letter vector to the console
        /// </summary>
        /// <param name="vector">the vector to print</param>
        public static void PrintLetter(double[] vector)
        {
            for (var i = 0; i < LetterHeight; i++)
            {
                for (var j = 0; j < LetterWidth; j++)
                    Console.Write(vector[i*10 + j] == 1 ? "*" : " ");
                Console.WriteLine("");
            }

            Console.WriteLine("");
        }

        /// <summary>
        /// print the letters from the matrix
        /// </summary>
        /// <param name="vectorDb">the vector db</param>
        /// <param name="numRows">the number of letters to print</param>
        /// <param name="startIdx">the starting index</param>
        public static void PrintLetters(double[][] vectorDb, int numRows, int startIdx = 0)
        {
            if (vectorDb == null) throw new ArgumentNullException("vectorDb");

            for (var letterIdx = startIdx; letterIdx < (numRows + startIdx) && letterIdx < vectorDb.Length; ++letterIdx)
            {
                PrintLetter(vectorDb[letterIdx]);
            }
        }

        /// <summary>
        /// Creates and returns vector db from the input letters located in the input folder
        /// </summary>
        /// <param name="printLetters">if true , prints each letter to the console</param>
        /// <returns>the vector db</returns>
        public static double[][] CreateVectorsDb(bool printLetters = false)
        {
            Console.WriteLine("Creating vector db from the letters in the input folder : " + LettersDirectoryPath);

            // get all the letters from the input directory
            string[] fileEntries = Directory.GetFiles(Path.GetFullPath(LettersDirectoryPath), FileTypes);

            // initialize the vectors db
            var allData = new double[fileEntries.GetLength(0)][];
            int currentIdx = 0;

            // for each letter, create it's represntation vector
            foreach (var fileEntry in fileEntries)
            {
                Console.WriteLine("Creating vector from image: " + fileEntry);

                // initialize it's bitmap
                var bitmap = new Bitmap(fileEntry);

                // check that the letter is from the correct size
                if (bitmap.Width != LetterWidth || bitmap.Height != LetterHeight)
                {
                    Console.WriteLine(fileEntry + " has incorrect dimensions. The dimension should be: " + LetterHeight + "X" +
                                      LetterWidth);
                    break;
                }

                // initialize the representation vector
                var representationVector = new double[RepresentationVectorSize + NumOfLabels];

                //get the pixel values
                for (var y = 0; y < bitmap.Height; y++)
                {
                    for (var x = 0; x < bitmap.Width; x++)
                    {
                        // get the current pixel
                        Color pixelColor = bitmap.GetPixel(x, y);

                        // if the pixel is not white set the value of the neuron to 1
                        if (pixelColor.A < PixelThreshold || pixelColor.B < PixelThreshold ||
                            pixelColor.G < PixelThreshold)
                            representationVector[y*10 + x] = 1;
                    }
                }

                // add the vector label from the name of the letter
                var fileName = Path.GetFileNameWithoutExtension(fileEntry);
                if (fileName != null && fileName.StartsWith("aleph"))
                    representationVector[RepresentationVectorSize + 0] = 1;
                else if (fileName != null && fileName.StartsWith("beith"))
                    representationVector[RepresentationVectorSize + 1] = 1;
                else if (fileName != null && fileName.StartsWith("giemel"))
                    representationVector[RepresentationVectorSize + 2] = 1;
                else if (fileName != null && fileName.StartsWith("daled"))
                    representationVector[RepresentationVectorSize + 3] = 1;
                else if (fileName != null && fileName.StartsWith("hei"))
                    representationVector[RepresentationVectorSize + 4] = 1;

                // add the representation vector to the db
                allData[currentIdx++] = representationVector;

                // print the current letter to the console
                if (printLetters) PrintLetters(allData, 1, currentIdx - 1);
            }

            // return the vector db
            return allData;
        }
   } 

} 