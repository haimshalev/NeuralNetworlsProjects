using System;
using System.Collections.Generic;
using System.Globalization;

namespace BamPhoneNumbersFrom16BitIcons.ConvertAlgo
{
    public interface IPhoneNumberToBiPolarConvertor
    {
        /// <summary>
        /// Convert a string phone number to biPolar format
        /// </summary>
        /// <param name="phoneNumber">input phone number</param>
        /// <returns>a bi Polar format of the phone number</returns>
        int[] ConvertStringPhoneNumberToBiPolar(string phoneNumber);

        /// <summary>
        /// Convert the bi Polar based phoneNumber to string
        /// </summary>
        /// <param name="biPolarPhoneNumber">biPolar PhoneNumber</param>
        /// <returns>string representation of the phone number</returns>
        string ConvertBiPolarPhoneNumberToString(int[] biPolarPhoneNumber);
    }

    public class PhoneNumberToBiPolarConvertorHuffmanCode : IPhoneNumberToBiPolarConvertor
    {
        private readonly IBinaryToBiPolarVecConvertor _binaryToBiPolarConvertor;

        public PhoneNumberToBiPolarConvertorHuffmanCode(IBinaryToBiPolarVecConvertor convertor)
        {
            _binaryToBiPolarConvertor = convertor;
        }

        /// <summary>
        /// Convert a string phone number to biPolar format
        /// </summary>
        /// <param name="phoneNumber">input phone number</param>
        /// <returns>a bi Polar format of the phone number</returns>
        public int[] ConvertStringPhoneNumberToBiPolar(string phoneNumber)
        {
            // Create an huffman code base conversation 
            // Every digit converted 0*digit and then 1
            var huffmanCode = new List<int>();

            foreach (var digit in phoneNumber)
            {
                var number = int.Parse(digit.ToString(CultureInfo.InvariantCulture));
                for (var i = 0; i < number; i++)
                    huffmanCode.Add(1);
                huffmanCode.Add(-1);
            }

            // need also to pad this to 100 with ones (10*1 + 10*9)
            var size = huffmanCode.Count;
            for (var i = size; i < 100; i++)
                huffmanCode.Add(-1);

            // return the array
            return huffmanCode.ToArray();
        }

        /// <summary>
        /// Convert the bi Polar based phoneNumber to string
        /// </summary>
        /// <param name="biPolarPhoneNumber">biPolar PhoneNumber</param>
        /// <returns>string representation of the phone number</returns>
        public string ConvertBiPolarPhoneNumberToString(int[] biPolarPhoneNumber)
        {
            // Convert from huffman code to string
            string phoneNumber = "";
            var huffmanString = string.Join("", _binaryToBiPolarConvertor.ConvertBiPolarVecToBinary(biPolarPhoneNumber));
            var digitArr = huffmanString.Split('0');
            for (var i = 0; i < 10; i++)
            {
                phoneNumber += digitArr[i].Length;
            }
            return phoneNumber;
        }
    }

    public class PhoneNumberToBiPolarConverorUsingBinaryDecode : IPhoneNumberToBiPolarConvertor
    {
        // 4 bits is enough for representing 10 numbers
        private const int BitsForDigit = 4;

        private readonly IBinaryToBiPolarVecConvertor _binaryToBiPolarConvertor;

        public PhoneNumberToBiPolarConverorUsingBinaryDecode(IBinaryToBiPolarVecConvertor convertor)
        {
            _binaryToBiPolarConvertor = convertor;
        }

        /// <summary>
        /// Convert a string phone number to biPolar format
        /// </summary>
        /// <param name="phoneNumber">input phone number</param>
        /// <returns>a bi Polar format of the phone number</returns>
        public int[] ConvertStringPhoneNumberToBiPolar(string phoneNumber)
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
            return _binaryToBiPolarConvertor.ConvertBinaryVecToBiPolar(binaryBuffer);
        }

        /// <summary>
        /// Convert the bi Polar based phoneNumber to string
        /// </summary>
        /// <param name="biPolarPhoneNumber">biPolar PhoneNumber</param>
        /// <returns>string representation of the phone number</returns>
        public string ConvertBiPolarPhoneNumberToString(int[] biPolarPhoneNumber)
        {
            // get the binary representation
            var binaryBuffer = _binaryToBiPolarConvertor.ConvertBiPolarVecToBinary(biPolarPhoneNumber);

            var phoneNumber = "";

            // convert the binary buffer to string
            for (var i = 0; i < binaryBuffer.Length; i += BitsForDigit)
            {
                int digit = 0;

                for (var j = BitsForDigit - 1; j >= 0; j--)
                    digit += binaryBuffer[i + j] * (int)(Math.Pow(2, BitsForDigit - 1 - j));

                if (digit > 9)
                    digit = 9;
                phoneNumber += digit;
            }

            return phoneNumber;
        }
    }
}
