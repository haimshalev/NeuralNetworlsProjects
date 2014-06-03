using System;
using System.Globalization;
using System.Threading;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace SelfOrgenizedMap
{

    public delegate void UpdateGuiDelegate(double[][] weights, double learnRate, int epoch);

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        private SelfOrgnizedMap _selfOrgenizedMap;
        private double[][] _data;
        private Thread _workingThread;

        public MainWindow()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Draw a point on the main canvas
        /// </summary>
        /// <param name="point">the point to draw. a double array of length 2</param>
        /// <param name="color">the color of the point</param>
        /// <param name="size">size of rect around the point</param>
        public void DrawPointOnCanvas(double[] point, Color color, int size)
        {
            // Draw rect
            var rect = new Rectangle
            {
                Stroke = new SolidColorBrush(color),
                Width = size,
                Height = size
            };
            Canvas.SetLeft(rect, point[0] - size/2 );
            Canvas.SetTop(rect, point[1] - size/2);

            MainCanvas.Children.Add(rect);
        }

        /// <summary>
        /// Draw a line on the main canvas
        /// </summary>
        /// <param name="point1">the start point of the line</param>
        /// <param name="point2">the end point of the line</param>
        /// <param name="color">the color of the brush of the line</param>
        /// <param name="size">the stroke thickness of the line</param>
        public void DrawLineOnCanvas(double[] point1, double[] point2, Brush color, int size)
        {
            var line = new Line
            {
                Stroke = color,
                X1 = point1[0],
                Y1 = point1[1],
                X2 = point2[0],
                Y2 = point2[1],
                StrokeThickness = size
            };
            MainCanvas.Children.Add(line);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="learnRate"></param>
        /// <param name="epochNum"></param>
        internal void UpdateWindow(double[][] weights, double learnRate, int epochNum)
        {
            if (!Dispatcher.CheckAccess())
            {
                // we were called on a worker thread
                // marshal the call to the user interface thread
                Dispatcher.Invoke(new UpdateGuiDelegate(UpdateWindow),
                            new object[] { weights, learnRate, epochNum});
                return;
            }

            // Clear the canvas from the previous update
            MainCanvas.Children.Clear();

            // Draw the data points
            foreach (var d in _data)
                DrawPointOnCanvas(d, Colors.Red, 1);

            // Draw all the neurons on the canvas
            foreach (var w in weights)
                DrawPointOnCanvas(w, Colors.Black, 3);

            // Draw the lines between neurons
            for (var i = 0; i < weights.Length - 1; i++)
                DrawLineOnCanvas(weights[i], weights[i + 1], Brushes.Navy, 1);

            // update the labels
            LearnRate.Text = learnRate.ToString(CultureInfo.InvariantCulture);
            EpochNum.Text = epochNum.ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Start learning on different thread
        /// </summary>
        /// <param name="data">the data to learn</param>
        /// <param name="nn">the SelfOrgenizedMap instance</param>
        private void StartWorking(double[][] data, SelfOrgnizedMap nn)
        {
            if (_workingThread != null) _workingThread.Abort();

            _data = data;
            _selfOrgenizedMap = nn;

            _selfOrgenizedMap.Data = _data;
            _selfOrgenizedMap.MainWindowRefreshRate = int.Parse(SetRefreshRate.Text);
            _selfOrgenizedMap.NumEpoches = int.Parse(SetNumEpoches.Text);

            // learn
            _workingThread = new Thread(_selfOrgenizedMap.Learn);
            _workingThread.Start();
        }

        /// <summary>
        /// Som neural network , line of neurons topology, uniform density data
        /// </summary>
        private void UnifromStartButton_Click(object sender, System.Windows.RoutedEventArgs e)
        {

            // prepare the data - ordered data
            var dataSize = (int)(MainCanvas.Height * MainCanvas.Width);
            
            var data = new double[dataSize][];
            for (var i = 0; i < MainCanvas.Height; i++)
                for (var j = 0; j < MainCanvas.Width; j++)
                {
                    data[(i * ((int)MainCanvas.Width)) + j] = new double[] { i, j };
                }

            // Initialize Self Orgenized map 
            var selfOrgenizedMap = new SelfOrgnizedMap(inputDimension: 2, numberOfClusters: 100, window: this);

            StartWorking(data, selfOrgenizedMap);
        }

        /// <summary>
        /// Som neural network, line of neurons topology, non uniform density data
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void NonUniformStartButton_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            var rand = new Random();

            // prepare the data - random data
            var dataSize = (int)(MainCanvas.Height * MainCanvas.Width);
            var data = new double[dataSize][];
            for (var i = 0; i < MainCanvas.Height; i++)
                for (var j = 0; j < MainCanvas.Width; j++)
                {
                    data[(i * ((int)MainCanvas.Width)) + j] = new []
                    {rand.Next((int) MainCanvas.Width/2), rand.Next((int) MainCanvas.Height/2) + MainCanvas.Height/2};
                }

            // Initialize Self Orgenized map 
            var selfOrgenizedMap = new SelfOrgnizedMap(inputDimension: 2, numberOfClusters: 100, window: this);

            StartWorking(data, selfOrgenizedMap);
        }
    }
}
