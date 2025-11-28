using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace SemanticLibrary.Wpf
{
    /// <summary>
    /// Interaction logic for SettingsWindow.xaml
    /// </summary>
    public partial class SettingsWindow : Window
    {
        public string PythonPath { get; set; }
        public string DatabasePath { get; set; }
        public string IndexPath { get; set; }
        public string Model { get; set; }
        public int TopK { get; set; }
        public double MinCosine { get; set; }
        public int FetchK { get; set; }

        public SettingsWindow()
        {
            InitializeComponent();
        }

        private void SettingsWindow_Loaded(object sender, RoutedEventArgs e)
        {
            PythonPathTextBox.Text = PythonPath ?? "python";
            DatabasePathTextBox.Text = DatabasePath ?? string.Empty;
            IndexPathTextBox.Text = IndexPath ?? string.Empty;

            ModelTextBox.Text = Model ?? "sentence-transformers/all-MiniLM-L6-v2";
            TopKTextBox.Text = TopK.ToString();
            MinCosTextBox.Text = MinCosine.ToString();
            FetchKTextBox.Text = FetchK.ToString();
        }

        private void FetchKTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void NumericOnly(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !Regex.IsMatch(e.Text, @"^[0-9\.\-]+$");
        }

        private void OkButton_Click(object sender, RoutedEventArgs e)
        {
            PythonPath = PythonPathTextBox.Text;
            Model = ModelTextBox.Text;
            int.TryParse(TopKTextBox.Text, out int topK);
            double.TryParse(MinCosTextBox.Text, out double minCosine);
            int.TryParse(FetchKTextBox.Text, out int fetchK);

            TopK = topK;
            MinCosine = minCosine;
            FetchK = fetchK;
            DialogResult = true;
            Close();
        }

        private void SelectPythonButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog()
            {
                Filter = "Python Executable (python.exe)|python.exe|All Executables (*.exe)|*.exe|All files (*.*)|*.*",
                Title = "Select Python Executable",
                FileName = "python.exe"
            };

            if (dialog.ShowDialog() == true)
            {
                PythonPath = dialog.FileName;
                PythonPathTextBox.Text = PythonPath;
            }
        }

        private void SelectDatabseButton_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog dialog = new SaveFileDialog()
            {
                Filter = "SQLite Database (*.db)|*.db|All files (*.*)|*.*",
                Title = "Select or create a library database",
                FileName = "library.db",
                OverwritePrompt = false
            };

            if (dialog.ShowDialog() == true)
            {
                DatabasePath = dialog.FileName;
                DatabasePathTextBox.Text = DatabasePath;
            }
        }

        private void SelectIndexButton_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog dialog = new SaveFileDialog()
            {
                Filter = "FAISS Index (*.faiss)|*.faiss|All files (*.*)|*.*",
                Title = "Select or create a FAISS index",
                FileName = "index.faiss",
                OverwritePrompt = false
            };

            if (dialog.ShowDialog() == true)
            {
                IndexPath = dialog.FileName;
                IndexPathTextBox.Text = IndexPath;
            }
        }
    }
}