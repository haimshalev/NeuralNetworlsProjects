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
        private readonly IPhoneNumberToBiPolarConvertor _phoneNumberToBiPolarConvertor;
        private readonly IBinaryToBiPolarVecConvertor _binaryToBiPolarVecConvertor;

        #endregion

        /// <summary>
        /// Cto'r
        /// </summary>
        public BamNetworkPhoneToIconsWrapper(IEnumerable<IconInputDataStructure> db, 
            IPhoneNumberToBiPolarConvertor phoneNumberToBiPolarConvertor,
            IBinaryToBiPolarVecConvertor binaryToBiPolarVecConvertor)
        {
            _phoneNumberToBiPolarConvertor = phoneNumberToBiPolarConvertor;
            _binaryToBiPolarVecConvertor = binaryToBiPolarVecConvertor;

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
            var biPolarIcon = _binaryToBiPolarVecConvertor.ConvertBinaryVecToBiPolar(iconInputDataStructure.IconVector);
            var biPolarPhoneNumber = _phoneNumberToBiPolarConvertor.ConvertStringPhoneNumberToBiPolar(iconInputDataStructure.PhoneNumber);

            // convert the input arguments to biPolar and run associate
            _bamNeuralNetwork.Associate(biPolarIcon, biPolarPhoneNumber);

            var phoneNumber = _phoneNumberToBiPolarConvertor.ConvertBiPolarPhoneNumberToString(biPolarPhoneNumber);
            var icon = _binaryToBiPolarVecConvertor.ConvertBiPolarVecToBinary(biPolarIcon);

            return new IconInputDataStructure(icon, phoneNumber);
        }

    }
}
