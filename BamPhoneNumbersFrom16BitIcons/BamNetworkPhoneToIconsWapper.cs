/****************************************
 * Neural Networks - Project No2 
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System.Collections.Generic;
using BamPhoneNumbersFrom16BitIcons.ConvertAlgo;

namespace BamPhoneNumbersFrom16BitIcons
{
    public class BamNetworkPhoneToIconsWrapper
    {
        #region Properties
       
        private readonly BamNeuralNetwork _bamNeuralNetwork;
        private readonly IPhoneNumberToBiPolarConvertor _phoneNumberToBiPolarConvertor = new PhoneNumberToBiPolarConvertorHuffmanCode(new BinaryToBiPolarVecConvertor());
        private readonly IBinaryToBiPolarVecConvertor _binaryToBiPolarVecConvertor = new BinaryToBiPolarVecConvertor();

        #endregion

        /// <summary>
        /// Cto'r
        /// </summary>
        public BamNetworkPhoneToIconsWrapper(IEnumerable<IconInputDataStructure> db)
        {
            // add the associations to the network
            foreach (var iconInputDataStructure in db)
            {
                var biPolarIcon = _binaryToBiPolarVecConvertor.ConvertBinaryVecToBiPolar(iconInputDataStructure.IconVector);
                var biPolarPhoneNumber = _phoneNumberToBiPolarConvertor.ConvertStringPhoneNumberToBiPolar(iconInputDataStructure.PhoneNumber);

                // Create the BAM neural network
                if (_bamNeuralNetwork == null)
                    _bamNeuralNetwork = new BamNeuralNetwork(biPolarIcon.Length, biPolarPhoneNumber.Length);    

                _bamNeuralNetwork.AddAssociation(biPolarIcon, biPolarPhoneNumber);
            }
        }

        /// <summary>
        /// Associates the input icon data structure to an icon which the networks belives in
        /// </summary>
        /// <param name="iconInputDataStructure">input icon</param>
        /// <returns>icon shich the network believes in</returns>
        public IconInputDataStructure Associate(IconInputDataStructure iconInputDataStructure)
        {
            var testVector = iconInputDataStructure.IconVector;
            var outputNumber = iconInputDataStructure.PhoneNumber;

            // Associate the new test vectors
            return Associate(testVector, outputNumber);
        }

        /// <summary>
        /// Associates the 2 vectors into an association which the network learned
        /// </summary>
        /// <param name="icon">16 bit icon</param>
        /// <param name="phoneNumber">string phone number</param>
        public IconInputDataStructure Associate(int[] icon, string phoneNumber)
        {
            var biPolarIcon = _binaryToBiPolarVecConvertor.ConvertBinaryVecToBiPolar(icon);
            var biPolarPhoneNumber = _phoneNumberToBiPolarConvertor.ConvertStringPhoneNumberToBiPolar(phoneNumber);

            // convert the input arguments to biPolar and run associate
            _bamNeuralNetwork.Associate(biPolarIcon, biPolarPhoneNumber);

            phoneNumber = _phoneNumberToBiPolarConvertor.ConvertBiPolarPhoneNumberToString(biPolarPhoneNumber);
            icon = _binaryToBiPolarVecConvertor.ConvertBiPolarVecToBinary(biPolarIcon);

            return new IconInputDataStructure(icon, phoneNumber);
        }

    }
}
