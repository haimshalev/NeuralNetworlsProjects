/****************************************
 * Neural Networks - Project No2 
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;
using System.Drawing;
using System.Text;

namespace BamPhoneNumbersFrom16BitIcons
{
    public class IconInputDataStructure
    {

        // each file size
        private const int IconHeight = 4;
        private const int IconWidth = 4;
        private const int RepresentationVectorSize = IconHeight * IconWidth;
        
        // icon vector
        public readonly int[] IconVector;
        public readonly string PhoneNumber;

        public IconInputDataStructure(int[] iconVector, string phoneNumber)
        {
            IconVector = new int[iconVector.Length];
            Array.Copy(iconVector, IconVector,iconVector.Length);
            PhoneNumber = phoneNumber;
        }

        public IconInputDataStructure(string fileEntry, string phoneNumber)
        {
            PhoneNumber = phoneNumber;

            // initialize it's bitmap
            var bitmap = new Bitmap(fileEntry);
            
            // initialize the representation vector
            IconVector = new int[RepresentationVectorSize];

            //get the pixel values
            for (var y = 0; y < bitmap.Height; y++)
            {
                for (var x = 0; x < bitmap.Width; x++)
                {
                    // get the current pixel
                    Color pixelColor = bitmap.GetPixel(x, y);

                    // if the pixel is not white set the value of the neuron to 1
                    if (pixelColor.GetBrightness() < 0.8)
                        IconVector[y * IconHeight + x] = 1;
                }
            }

        }

        /// <summary>
        /// return a presentation of the icon and the associated phone number
        /// </summary>
        /// <returns>string representation of the icon and the phone number </returns>
        public override string ToString()
        {
            return ToString(0);
        }

        /// <summary>
        /// return a presentation of the icon and the associated phone number
        /// </summary>
        /// <returns>string representation of the icon and the phone number </returns>
        public string ToString(int numOfTabs)
        {
            var tabs = "";
            for (var i = 0; i < numOfTabs; i++)
                tabs += "\t";

            var str = new StringBuilder(tabs + "Icon : \n" + tabs + "\t");

            for (var i = 0; i < IconHeight; i++)
            {
                for (var j = 0; j < IconWidth; j++)
                    str.Append(IconVector[i * IconHeight + j] == 1 ? "*" : " ");
                str.AppendLine();
                str.Append(tabs + "\t");
            }

            str.AppendLine();

            str.Append(tabs + "PhoneNumber : ");
            str.Append(PhoneNumber);
            str.AppendLine();

            return str.ToString();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="errorPrecentage"></param>
        /// <returns></returns>
        public IconInputDataStructure CreateError(int errorPrecentage)
        {
            var errorIcon = new IconInputDataStructure(IconVector, PhoneNumber);
            var numOfPixelsToChange = (int) (RepresentationVectorSize*((double) errorPrecentage/100));
            var rand = new Random(DateTime.Now.Millisecond);

            for (var i = 0; i < numOfPixelsToChange; i++)
            {
                // Get a random index and change it's value
                var index = rand.Next(0, RepresentationVectorSize - 1);

                // Flip bit
                errorIcon.IconVector[index] = 1 - errorIcon.IconVector[index];
            }

            return errorIcon;
        }
    }
}
