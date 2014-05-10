using System;
using System.Globalization;

namespace BamPhoneNumbersFrom16BitIcons
{
    class BamNetworkPhoneToIconsWrapper
    {
        // 4 bits is enough for representing 10 numbers
        private const int BitsForDigit = 4;

        #region Properties
        
            private string[] _phoneNumbers;
            private int[][] _biPolarPhoneNumbers;
            private int[][] _inputIcons;
            private int[][] _biPolarIcons;

            private int _numberInputNeurons;
            private int _numberOutputNeurons;

            private readonly BamNeuralNetwork _bamNeuralNetwork; 

        #endregion

        /// <summary>
        /// Cto'r
        /// </summary>
        /// <param name="icons">a list of icons to add to add to the network</param>
        /// <param name="phoneNumbers">a list of phone numbers to add to the network</param>
        public BamNetworkPhoneToIconsWrapper(int[][] icons, string[] phoneNumbers)
        {
            //Some sanity checks
            if (icons == null || phoneNumbers == null || icons[0] == null || phoneNumbers[0] == null)
                throw new ArgumentException("input vectors must contain values");

            // store the input associations
            _inputIcons = icons;
            _phoneNumbers = phoneNumbers;

            // convert the associations to biPolar format
            _biPolarPhoneNumbers = ConvertStringPhoneNumbersToBiPolar(phoneNumbers);
            _biPolarIcons = Convert16BitIconsToBiPolar(icons);

            // set the size of the inner neural network
            _numberInputNeurons = _biPolarIcons[0].Length;
            _numberOutputNeurons = _biPolarPhoneNumbers[0].Length;

            // Create the BAM neural network
            _bamNeuralNetwork = new BamNeuralNetwork(_numberInputNeurons, _numberOutputNeurons);

            // add the associations to the network
            for (var i = 0; i < _biPolarIcons.Length; i++)
            {
                // Test the convert methods
                Console.WriteLine(_phoneNumbers[i]);
                Console.WriteLine(ConvertBiPolarPhoneNumberToString(_biPolarPhoneNumbers[i]));

                _bamNeuralNetwork.AddAssociation(_biPolarIcons[i], _biPolarPhoneNumbers[i]);
            }
        }

        /// <summary>
        /// Convert the string phone numbers into a biPolar format
        /// </summary>
        /// <param name="phoneNumbers">string array of phone numbers</param>
        /// <returns>a biPolar phoneNumber array</returns>
        public static int[][] ConvertStringPhoneNumbersToBiPolar(string[] phoneNumbers)
        {
            // initialize a new biPolar phonenumber list
            var biPolarPhoneNumbers = new int[phoneNumbers.Length][];

            // for each icon in the input argument 
            for (var i = 0; i < phoneNumbers.Length; i++)
            {
                // add it to the icon list
                biPolarPhoneNumbers[i] = ConvertStringPhoneNumberToBiPolar(phoneNumbers[i]);
            }

            return biPolarPhoneNumbers;
        }

        /// <summary>
        /// Convert a string phone number to biPolar format
        /// </summary>
        /// <param name="phoneNumber">input phone number</param>
        /// <returns>a bi Polar format of the phone number</returns>
        private static int[] ConvertStringPhoneNumberToBiPolar(string phoneNumber)
        {
            // initialize a binary buffer
            var binaryBuffer = new int[phoneNumber.Length * BitsForDigit];

            var i = 0;

            // for each digit , convert it to binary
            foreach (var digit in phoneNumber)
            {
                // parse the digit to number
                var x = int.Parse(digit.ToString(CultureInfo.InvariantCulture));

                i += BitsForDigit;
                var iteration = 1;
                
                // add the ones where needed
                while (x != 0)
                {
                    if ((x & 1) == 1)
                        binaryBuffer[i - iteration] = 1;
                    
                    x >>= 1;
                    iteration++;
                }

            }
            
            // return the biPolar Representation
            return Convert16BitIconToBiPolar(binaryBuffer);
        }

        /// <summary>
        /// Convert the bi Polar based phoneNumber to string
        /// </summary>
        /// <param name="biPolarPhoneNumber">biPolar PhoneNumber</param>
        /// <returns>string representation of the phone number</returns>
        public static string ConvertBiPolarPhoneNumberToString(int[] biPolarPhoneNumber)
        {
            // get the binary representation
            var binaryBuffer = Convert16BitBiPolarIconToBinary(biPolarPhoneNumber);

            var phoneNumber = "";

            // convert the binary buffer to string
            for (var i = 0; i < binaryBuffer.Length; i += BitsForDigit)
            {
                int digit = 0;

                for (var j = BitsForDigit - 1; j >= 0; j--)
                    digit += binaryBuffer[i + j]* (int)(Math.Pow(2, BitsForDigit -1 - j));

                phoneNumber += digit;
            }

            return phoneNumber;

        }

        /// <summary>
        /// Convert the boolean icons into a biPolar format
        /// </summary>
        /// <param name="icons">16 bit arrays of 1 and 0</param>
        /// <returns>a biPolar icons array</returns>
        public static int[][] Convert16BitIconsToBiPolar(int[][] icons)
        {
            // initialize a new biPolar icons list
            var biPolarIcons = new int[icons.Length][];

            // for each icon in the input argument 
            for (var i = 0; i < icons.Length; i++)
            {
                // add it to the icon list
                biPolarIcons[i] = Convert16BitIconToBiPolar(icons[i]);
            }

            return biPolarIcons;
        }

        /// <summary>
        /// Convert the boolean binaryt based icon to biPolar format
        /// </summary>
        /// <param name="icon">input binary icon</param>
        /// <returns>a biPolar icon</returns>
        public static int[] Convert16BitIconToBiPolar(int[] icon)
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
        public static int[] Convert16BitBiPolarIconToBinary(int[] biPolarIcon)
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

        /// <summary>
        /// Associates the 2 vectors into an association which the network learned
        /// </summary>
        /// <param name="icon">16 bit icon</param>
        /// <param name="phoneNumber">string phone number</param>
        public void Associate(int[] icon, string phoneNumber)
        {
            // convert the input arguments to biPolar and run associate
            Associate(Convert16BitIconToBiPolar(icon), ConvertStringPhoneNumberToBiPolar(phoneNumber));
        }

        /// <summary>
        /// Associates the 2 vectors into an association which the network learned
        /// </summary>
        /// <param name="biPolarIcon">biPolar format of an icon</param>
        /// <param name="biPolarPhoneNumber">biPolar format of a phoneNumber</param>
        public void Associate(int[] biPolarIcon, int[] biPolarPhoneNumber)
        {
            _bamNeuralNetwork.Associate(biPolarIcon, biPolarPhoneNumber);
        }
    }
}
