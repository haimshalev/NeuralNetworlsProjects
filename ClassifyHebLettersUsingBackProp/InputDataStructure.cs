/****************************************
 * Neural Networks - Project No1 
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;
using System.Drawing;
using System.IO;
using System.Text;

namespace ClassifyHebLettersUsingBackProp
{

    /// <summary>
    /// Base class for neural network data
    /// </summary>
    public class InputDataStructure
    {
        public double[] DataVector;
        public double[] TargetVector;

        // Cto'r
        public InputDataStructure(int dataLength, int targetLength)
        {
            DataVector = new double[dataLength];
            TargetVector = new double[targetLength];
        }

        /// <summary>
        /// Get a copy of this instance
        /// </summary>
        public virtual InputDataStructure GetCopy()
        {
            var copy = new InputDataStructure(DataVector.Length,TargetVector.Length);
            Array.Copy(DataVector, copy.DataVector, DataVector.Length);
            Array.Copy(TargetVector, copy.TargetVector, DataVector.Length);
            return copy;
        }

        /// <summary>
        /// Returns a new copy of the Data vector
        /// </summary>
        public double[] GetDataVectorCopy()
        {
            var copyVector = new double[DataVector.Length];
            Array.Copy(DataVector, copyVector, DataVector.Length);
            return copyVector;
        }

        /// <summary>
        /// Returns a new copy of the target vector
        /// </summary>
        public double[] GetTargetVectorCopy()
        {
            var copyVector = new double[TargetVector.Length];
            Array.Copy(TargetVector, copyVector, TargetVector.Length);
            return copyVector;
        }
    }


    /// <summary>
    /// Heb letter neural network input class
    /// </summary>
    class HebLetterInputDataStructure : InputDataStructure
    {

        // each file size
        private const int LetterHeight = 10;
        private const int LetterWidth = 10;
        private const int RepresentationVectorSize = LetterHeight * LetterWidth;

        // for now we are labeling only 5 letter - aleph, beith, giemel, daled, hei
        private const int NumOfLabels = 5;

        // the threshold for a specific pixel to be white
        private const int PixelThreshold = 200;

        protected HebLetterInputDataStructure(int dataLength, int targetLength) : base(dataLength, targetLength)
        {
        }

        public HebLetterInputDataStructure(string fileEntry) : base(RepresentationVectorSize, NumOfLabels)
        {
            Console.WriteLine("Creating vector from image: " + fileEntry);

            // initialize it's bitmap
            var bitmap = new Bitmap(fileEntry);

            // check that the letter is from the correct size
            if (bitmap.Width != LetterWidth || bitmap.Height != LetterHeight)
            {
                throw new ArgumentOutOfRangeException(fileEntry + " has incorrect dimensions. The dimension should be: " + LetterHeight + "X" +
                                  LetterWidth);
            }

            // initialize the representation vector
            DataVector = new double[RepresentationVectorSize];
            TargetVector = new double[NumOfLabels];

            //get the pixel values
            for (var y = 0; y < bitmap.Height; y++)
            {
                for (var x = 0; x < bitmap.Width; x++)
                {
                    // get the current pixel
                    Color pixelColor = bitmap.GetPixel(x, y);

                    // if the pixel is not white set the value of the neuron to 1
                    if (pixelColor.GetBrightness() < 0.8)
                        DataVector[y * 10 + x] = 1;
                }
            }

            // add the vector label from the name of the letter
            var fileName = Path.GetFileNameWithoutExtension(fileEntry);
            if (fileName != null && fileName.StartsWith("aleph"))
                TargetVector[0] = 1;
            else if (fileName != null && fileName.StartsWith("beith"))
                TargetVector[1] = 1;
            else if (fileName != null && fileName.StartsWith("giemel"))
                TargetVector[2] = 1;
            else if (fileName != null && fileName.StartsWith("daled"))
                TargetVector[3] = 1;
            else if (fileName != null && fileName.StartsWith("hei"))
                TargetVector[4] = 1;

        }

        public override InputDataStructure GetCopy()
        {
            var copy = new HebLetterInputDataStructure(DataVector.Length, TargetVector.Length);
            Array.Copy(DataVector, copy.DataVector, DataVector.Length);
            Array.Copy(TargetVector, copy.TargetVector, TargetVector.Length);
            return copy;
        }

        /// <summary>
        /// return a presentation of the letter
        /// </summary>
        /// <returns>string representation of the letter</returns>
        public override string ToString()
        {
            var str = new StringBuilder();

            for (var i = 0; i < LetterHeight; i++)
            {
                for (var j = 0; j < LetterWidth; j++)
                    str.Append(DataVector[i * 10 + j] == 1 ? "*" : " ");
                str.Append("\n");
            }

            str.Append("\n");
            return str.ToString();
        }
    }
}
