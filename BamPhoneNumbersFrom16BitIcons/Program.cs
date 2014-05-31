/****************************************
 * Neural Networks - Project No2 
 * 
 * ZahiKfir         200681476
 * Haim Shalalevili 200832780
 * Nadav Eichler    308027325
 ***************************************/

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BamPhoneNumbersFrom16BitIcons
{
    class BamPhoneNumbersFrom16BitIconsProgram
    {

        // input icons directory
        private const string IconsDirectoryPath = @"icons";

        // file types - for now using only png files
        private const string FileTypes = "*.bmp";

        static void Main()
        {
            Console.WriteLine("********************************************");
            Console.WriteLine("Neural Networks - Project No.2 - BAM network");
            Console.WriteLine("********************************************");

            #region Prepare the data

            // initialize output vectors - phone numbers
            var phoneNumbers = new[] { "3120440569", "9114567896", "0538795012", "1234567890", "0508656789" };

            // initializing input vectors - 16 bit icons
            var db = ReadIcons(phoneNumbers);

            Console.WriteLine("\nAssociations Icons: \n");
            foreach (var icon in db)
                Console.WriteLine(icon.ToString(1));

            #endregion

            // initialize the BAM neural network wrapper
            var bamNeuralNetworkWrapper = new BamNetworkPhoneToIconsWrapper(db);

            Console.WriteLine("Test BAM Network : \n");
            
            TestInputIcons(db, bamNeuralNetworkWrapper);
            TestInputIconsWithPrecentError(db, bamNeuralNetworkWrapper, 20);
            TestInputIconsWithPrecentError(db, bamNeuralNetworkWrapper, 50);
        }

        /// <summary>
        /// Read the icons from the iconsDirectory and associate them with a phone number
        /// </summary>
        /// <param name="phoneNumbers">array of phone numbers to associate to icons</param>
        /// <returns>list of icons attached to phone number</returns>
        private static List<IconInputDataStructure> ReadIcons(string[] phoneNumbers)
        {
            Console.WriteLine("\nCreating vector db from the letters in the input folder : " + IconsDirectoryPath);

            // get all the letters from the input directory
            string[] fileEntries = Directory.GetFiles(Path.GetFullPath(IconsDirectoryPath), FileTypes);

            var i = 0;

            // for each letter, create it's represntation vector
            return fileEntries.Select(fileEntry => new IconInputDataStructure(fileEntry, phoneNumbers[i++])).ToList();
        }

        #region TestMethods

        private static void TestInputIcons(IEnumerable<IconInputDataStructure> db, BamNetworkPhoneToIconsWrapper bamNeuralNetworkWrapper)
        {
            Console.WriteLine("Test exactly the input icons:  ");

            // check all the input icons
            foreach (var iconInputDataStructure in db)
            {
                Console.ReadKey();
                Console.WriteLine("\tInputIcon: ");
                Console.WriteLine(iconInputDataStructure.ToString(2));

                // output results
                Console.WriteLine("\tOutputIcon:");
                Console.WriteLine( bamNeuralNetworkWrapper.Associate(iconInputDataStructure).ToString(2));
            }
        }

        private static void TestInputIconsWithPrecentError(IEnumerable<IconInputDataStructure> db,
            BamNetworkPhoneToIconsWrapper bamNeuralNetworkWrapper, int errorPercentage)
        {
            Console.WriteLine("Test the input icons with " + errorPercentage + "% random Error");

            // check all the input icons
            foreach (var iconInputDataStructure in db)
            {
                Console.ReadKey();
                Console.WriteLine("\tInputIcon: ");
                Console.WriteLine(iconInputDataStructure.ToString(2));

                // create 5 random 20% error icons and test the result
                for (var i = 0; i < 5; i++)
                {
                    Console.ReadKey();
                    Console.WriteLine("\t\t" + errorPercentage +  "% random error icon:");
                    var errorIcon = iconInputDataStructure.CreateError(errorPercentage);
                    Console.WriteLine(errorIcon.ToString(3));

                    // output results
                    Console.WriteLine("\t\tOutputIcon:");
                    Console.WriteLine(bamNeuralNetworkWrapper.Associate(errorIcon).ToString(3));
                }
            }
        }

        #endregion
    }
}


