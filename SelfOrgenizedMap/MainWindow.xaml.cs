using System.Diagnostics.Eventing.Reader;
using System.Threading;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace SelfOrgenizedMap
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        public MainWindow()
        {
            InitializeComponent();

            // Initialize Self Orgenized map 
            var selfOrgenizedMap = new SelfOrgnizedMap(2, 10, this);

            // prepare the data

            // learn
            selfOrgenizedMap.Learn(inputData, 100, 5);
        }


        public void DrawOnCanvas(Color color, int x, int y)
        {
            // Draw rect
            var rect = new Rectangle
            {
                Stroke = new SolidColorBrush(color),
                Width = 15,
                Height = 15
            };
            Canvas.SetLeft(rect, x);
            Canvas.SetTop(rect, y);
            MainCanvas.Children.Add(rect);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="weights"></param>
        internal void UpdateWindow(double[][] weights)
        {
            // Draw all the neurons on the canvas
            foreach (var w in weights)
                DrawOnCanvas(Colors.Black, (int)w[0], (int)w[1]);

            // stoll the gui for the user to see results
            // TODO: replace it by a button press
            Thread.Sleep(1000);
        }
    }
}
