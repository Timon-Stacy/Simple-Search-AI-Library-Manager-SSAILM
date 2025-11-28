using AILibrary;
using Microsoft.Win32;
using Newtonsoft.Json;
using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows;
using System.Collections.Generic;
using System.IO;

namespace SemanticLibrary.Wpf
{
    public partial class DownloadWindow : Window
    {
        private ObservableCollection<DownloadClass> _downloads = new ObservableCollection<DownloadClass>();

        public DownloadWindow()
        {
            InitializeComponent();

            _downloads.Add(new DownloadClass());   // first blank row
            DownloadDataGrid.ItemsSource = _downloads;
        }

        private async Task DownloadAsync()
        {
            try
            {
                var main = this.Owner as MainWindow;

                // Get settings from config
                string pythonPath = "python";
                string databasePath = "library.db";

                try
                {
                    Utilities.LoadConfig(
                        out pythonPath,
                        out databasePath,
                        out _,
                        out _,
                        out _,
                        out _,
                        out _);
                }
                catch
                {
                    pythonPath = "python";
                    databasePath = "library.db";
                }

                string downloadScript = Utilities.GetScriptPath("download.py");

                await DownloadUtil.DownloadAsync(
                    _downloads,
                    pythonPath,
                    downloadScript,
                    databasePath,
                    line =>
                    {
                        // log to MainWindow only
                        if (main != null)
                        {
                            main.Dispatcher.Invoke(() =>
                                main.AppendLog("[Download] " + line));
                        }
                    });

                MessageBox.Show("Download finished.");
                this.DialogResult = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show("An error occurred during download: " + ex.Message);
            }
            finally
            {
                DownloadProgressBar.Visibility = Visibility.Hidden;
            }
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            DownloadProgressBar.Visibility = Visibility.Visible;
            await DownloadAsync();
        }

        private void ImportButton_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Title = "Select download list",
                Filter = "Book list (*.json)|*.json|All files (*.*)|*.*"
            };

            if (dlg.ShowDialog() == true)
            {
                try
                {
                    string json = File.ReadAllText(dlg.FileName);

                    // Deserialize into a list of DownloadClass
                    var items = JsonConvert.DeserializeObject<List<DownloadClass>>(json)
                                ?? new List<DownloadClass>();

                    _downloads.Clear();

                    foreach (var item in items)
                    {
                        // basic validation so we don't send garbage to Python
                        if (string.IsNullOrWhiteSpace(item.Url) ||
                            string.IsNullOrWhiteSpace(item.Title))
                        {
                            continue; // skip invalid rows
                        }

                        if (string.IsNullOrWhiteSpace(item.Author))
                            item.Author = "Unknown";

                        if (string.IsNullOrWhiteSpace(item.Category))
                            item.Category = "Uncategorized";

                        _downloads.Add(item);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Failed to load download list: " + ex.Message,
                                    "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

    }
}