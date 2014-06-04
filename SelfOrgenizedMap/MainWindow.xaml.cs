using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Shapes;

namespace SelfOrgenizedMapNamespace
{

    public delegate void UpdateGuiDelegate(double[][] weights, List<KeyValuePair<int, int>> neighborhoodList, double learnRate, int epoch, int iterationNum);

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        private SelfOrgenizedMap _selfOrgenizedMap;
        private double[][] _data;
        private Thread _workingThread;
        private bool _firstRun = false;

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
        /// update the ui with the information about the current state
        /// </summary>
        /// <param name="weights">current nn weights (kohonen neuron location)</param>
        /// <param name="neighborhoodList">list of neighboors acording to the net topology</param>
        /// <param name="learnRate">current learn rate</param>
        /// <param name="epochNum">current epoch round</param>
        /// <param name="iterationNum">current learning iteration</param>
        internal void UpdateWindow(double[][] weights, List<KeyValuePair<int, int>>neighborhoodList, double learnRate, int epochNum, int iterationNum)
        {

            if (!Dispatcher.CheckAccess())
            {
                Thread.Sleep(500);

                // we were called on a worker thread
                // marshal the call to the user interface thread
                Dispatcher.Invoke(new UpdateGuiDelegate(UpdateWindow),
                            new object[] { weights, neighborhoodList, learnRate, epochNum, iterationNum});
                return;
            }

            // Draw the data points once
            if (_firstRun)
            {
                MainCanvas.Children.Clear();

                _firstRun = false;
                foreach (var d in _data)
                    DrawPointOnCanvas(d, Colors.Red, 1);
            }
            else
            {
                // Clear the canvas from the previous update
                MainCanvas.Children.RemoveRange(_data.Length, MainCanvas.Children.Count);
            }
            

            // Draw all the neurons on the canvas
            foreach (var w in weights)
                DrawPointOnCanvas(w, Colors.Black, 3);

            // Draw the lines between neurons
            foreach (var keyValuePair in neighborhoodList)
                DrawLineOnCanvas(weights[keyValuePair.Key], weights[keyValuePair.Value], Brushes.Navy, 1);

            // update the labels
            LearnRate.Text = learnRate.ToString(CultureInfo.InvariantCulture);
            EpochNum.Text = epochNum.ToString(CultureInfo.InvariantCulture);
            IterationNum.Text = iterationNum.ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Start learning on different thread
        /// </summary>
        /// <param name="data">the data to learn</param>
        /// <param name="nn">the SelfOrgenizedMap instance</param>
        private void StartWorking(double[][] data, SelfOrgenizedMap nn)
        {
            if (_workingThread != null) _workingThread.Abort();

            _data = data;
            _selfOrgenizedMap = nn;

            _selfOrgenizedMap.Data = _data;
            _selfOrgenizedMap.MainWindowRefreshRate = int.Parse(SetRefreshRate.Text);
            _selfOrgenizedMap.NumEpoches = int.Parse(SetNumEpoches.Text);

            _firstRun = true;

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
            var selfOrgenizedMap = new SelfOrgnizedMap<LineTopology>(2, int.Parse(SetNumOfClasters.Text), this);

            StartWorking(data, selfOrgenizedMap);
        }

        /// <summary>
        /// Som neural network, line of neurons topology, non uniform density data
        /// </summary>
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
            var selfOrgenizedMap = new SelfOrgnizedMap<LineTopology>(2, int.Parse(SetNumOfClasters.Text), this);

            StartWorking(data, selfOrgenizedMap);
        }

        /// <summary>
        /// Som neural network, circle of neurons topology, torus data
        /// </summary>
        private void TorusStartButton_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            // prepare the data - create a torus 
            var data = new List<double[]>();

            const int R = 40;
            const int r = 10;
            for (var i = 0; i < 360; i = i + 3)
                for (var j = 0; j < 360; j = j +3)
                {
                    double radians1 = i/(180/Math.PI);
                    double radians2 = j/(180/Math.PI);
                    double x = (R + r * Math.Cos(radians1)) * Math.Cos(radians2);
                    double y = (R + r * Math.Cos(radians1)) * Math.Sin(radians2);

                    data.Add(new []{x + 50 ,y + 50});
                }


            // Initialize Self Orgenized map 
            var selfOrgenizedMap = new SelfOrgnizedMap<CircleTopology>(2, int.Parse(SetNumOfClasters.Text), this);

            StartWorking(data.ToArray(), selfOrgenizedMap);
        }
    }
}
