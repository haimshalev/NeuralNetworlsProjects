using System;

namespace BamPhoneNumbersFrom16BitIcons
{
    class BamNetworkPhoneToIconsWrapper
    {
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
            for (var i = 0; i < _biPolarIcons.Length ; i++)
                _bamNeuralNetwork.AddAssociation(_biPolarIcons[i], _biPolarPhoneNumbers[i]);
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
            throw new NotImplementedException();
        }

        /// <summary>
        /// Convert the bi Polar based phoneNumber to string
        /// </summary>
        /// <param name="biPolarPhoneNumber">biPolar PhoneNumber</param>
        /// <returns>string representation of the phone number</returns>
        public static string ConvertBiPolarPhoneNumberToString(int[] biPolarPhoneNumber)
        {
            throw new NotImplementedException();
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
            throw new NotImplementedException();            
        }
        
        /// <summary>
        /// Convert the biPolar based icon to binary format
        /// </summary>
        /// <param name="icon">biPolar icon</param>
        /// <returns>a binary icon</returns>
        public static int[] Convert16BitBiPolarIconToBinary(int[] icon)
        {
            throw new NotImplementedException();            
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
