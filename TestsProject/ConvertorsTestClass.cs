using BamPhoneNumbersFrom16BitIcons.ConvertAlgo;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TestsProject
{
    [TestClass]
    public class ConvertorsTestClass
    {
        [TestMethod]
        public void TestPhoneNumberToIntArrConverter()
        {
            ////Test the converter
            //string phoneNumber = "0123456789";
            
            //var huffmanCode = BamNetworkPhoneToIconsWrapper.ConvertStringPhoneNumberToBiPolar(phoneNumber);
        }

        [TestMethod]
        public void TestIntArrToPhoneNumberConverter()
        {
            const string requiredPhoneNumber = "0123456789";
            var intArr = new[] {-1,1,-1,1,1,-1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,1};

            IPhoneNumberToBiPolarConvertor convertor = 
                new PhoneNumberToBiPolarConvertorHuffmanCode(new BinaryToBiPolarVecConvertor());

            var phoneNumber = convertor.ConvertBiPolarPhoneNumberToString(intArr);

            Assert.AreEqual(phoneNumber, requiredPhoneNumber);
        }
    }
}
