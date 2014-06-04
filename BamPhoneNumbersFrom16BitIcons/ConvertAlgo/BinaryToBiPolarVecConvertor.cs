/****************************************
 * Neural Networks - Project No2
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;

namespace BamPhoneNumbersFrom16BitIcons.ConvertAlgo
{
    public interface IBinaryToBiPolarVecConvertor
    {
        /// <summary>
        /// Convert the boolean binaryt based icon to biPolar format
        /// </summary>
        /// <param name="icon">input binary icon</param>
        /// <returns>a biPolar icon</returns>
        int[] ConvertBinaryVecToBiPolar(int[] icon);

        /// <summary>
        /// Convert the biPolar based icon to binary format
        /// </summary>
        /// <param name="biPolarIcon">biPolar icon</param>
        /// <returns>a binary icon</returns>
        int[] ConvertBiPolarVecToBinary(int[] biPolarIcon);
    }

    public class BinaryToBiPolarVecConvertor : IBinaryToBiPolarVecConvertor
    {
        /// <summary>
        /// Convert the boolean binaryt based icon to biPolar format
        /// </summary>
        /// <param name="icon">input binary icon</param>
        /// <returns>a biPolar icon</returns>
        public int[] ConvertBinaryVecToBiPolar(int[] icon)
        {
            // initialize the biPolar buffer
            var biPolarBuffer = new int[icon.Length];

            for (var i = 0; i < icon.Length; i++)
            {
                switch (icon[i])
                {
                    case 1:
                        biPolarBuffer[i] = 1;
                        break;
                    case 0:
                        biPolarBuffer[i] = -1;
                        break;
                    default:
                        throw new Exception("the icon has wrong bit value");
                }
            }

            return biPolarBuffer;
        }

        /// <summary>
        /// Convert the biPolar based icon to binary format
        /// </summary>
        /// <param name="biPolarIcon">biPolar icon</param>
        /// <returns>a binary icon</returns>
        public int[] ConvertBiPolarVecToBinary(int[] biPolarIcon)
        {
            // initialize the biPolar buffer
            var binaryBuffer = new int[biPolarIcon.Length];

            for (var i = 0; i < biPolarIcon.Length; i++)
            {
                switch (biPolarIcon[i])
                {
                    case 1:
                        binaryBuffer[i] = 1;
                        break;
                    case -1:
                        binaryBuffer[i] = 0;
                        break;
                    default:
                        throw new Exception("the icon has wrong bit value");
                }
            }

            return binaryBuffer;
        }
    }
}
