using AILibrary;
using Microsoft.Win32;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Newtonsoft.Json;


namespace SemanticLibrary.Wpf
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string _pythonPath;
        private string _databasePath;
        private string _indexPath;
        private List<SearchResult> _results;
        private string _model = "sentence-transformers/all-MiniLM-L6-v2";
        private int _topK = 5;
        private double _minCosine = -1.0;
        private int _fetchK = 200;
        private string _query = "";

        public MainWindow()
        {
            InitializeComponent();

            _pythonPath = "python";
            _databasePath = string.Empty;
            _indexPath = string.Empty;
            _model = "sentence-transformers/all-MiniLM-L6-v2";
            _topK = 5;
            _minCosine = -1.0;
            _fetchK = 200;

            try
            {
                Utilities.LoadConfig(
                    out _pythonPath,
                    out _databasePath,
                    out _indexPath,
                    out _model,
                    out _topK,
                    out _minCosine,
                    out _fetchK);
            }
            catch
            {
                _pythonPath = "python";
                _databasePath = string.Empty;
                _indexPath = string.Empty;
            }

            try
            {
                if (!string.IsNullOrWhiteSpace(_databasePath) &&
                    File.Exists(_databasePath))
                {
                    var table = Utilities.LoadData(_databasePath);
                    LibraryDataGrid.ItemsSource = table?.DefaultView;
                }
                else
                {
                    LibraryDataGrid.ItemsSource = null;
                }
            }
            catch
            {
                LibraryDataGrid.ItemsSource = null;
            }
        }

        public void AppendLog(string line)
        {
            LogListBox.Items.Add(line);
            LogListBox.ScrollIntoView(LogListBox.Items[LogListBox.Items.Count - 1]);
        }

        private async void AISearch()
        {
            try
            {
                string searchScript = Utilities.GetScriptPath("search.py");
                _results = await Task.Run(() => Search.PythonSearch(
                    _pythonPath,
                    searchScript,
                    _query,
                    _topK,
                    _minCosine,
                    _fetchK,
                    _model,
                    _databasePath,
                    _indexPath));
                ResultsDataGrid.ItemsSource = _results;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                SearchProgressBar.Visibility = Visibility.Hidden;
            }
        }

        private void SearchButton_Click(object sender, RoutedEventArgs e)
        {
            SearchProgressBar.Visibility = Visibility.Visible;
            _query = SearchTextBox.Text;
            AISearch();
        }

        private void ResultsDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ResultsDataGrid.SelectedItems.Count == 0)
                return;

            var selected = ResultsDataGrid.SelectedItems;
            foreach (SearchResult row in selected)
            {
                ResultsTextBox.Text = row.text;
            }
        }

        private void SettingsButton_Click(object sender, RoutedEventArgs e)
        {
            var settings = new SettingsWindow
            {
                PythonPath = _pythonPath,
                DatabasePath = _databasePath,
                IndexPath = _indexPath,
                Model = _model,
                TopK = _topK,
                FetchK = _fetchK,
                MinCosine = _minCosine
            };

            bool? result = settings.ShowDialog();

            if (result == true)
            {
                _pythonPath = settings.PythonPath;
                _databasePath = settings.DatabasePath;
                _indexPath = settings.IndexPath;
                _model = settings.Model;
                _topK = settings.TopK;
                _fetchK = settings.FetchK;
                _minCosine = settings.MinCosine;
            }

            Utilities.SaveConfig(_pythonPath, _databasePath, _indexPath, _model, _topK, _minCosine, _fetchK);

            try
            {
                if (!string.IsNullOrWhiteSpace(_databasePath) &&
                    File.Exists(_databasePath))
                {
                    var table = Utilities.LoadData(_databasePath);
                    LibraryDataGrid.ItemsSource = table?.DefaultView;
                }
                else
                {
                    LibraryDataGrid.ItemsSource = null;
                }
            }
            catch
            {
                LibraryDataGrid.ItemsSource = null;
            }
        }

        private async Task EmbedAsync()
        {
            try
            {
                string embedScript = Utilities.GetScriptPath("embed_library.py");
                await Task.Run(() =>
                {
                    EmbedUtil.Embed(_pythonPath, embedScript, _databasePath, _indexPath, line =>
                    {
                        // Safely marshal back to UI thread
                        Dispatcher.Invoke(() =>
                        {
                            LogListBox.Items.Add(line);

                            // Auto-scroll to newest item
                            LogListBox.ScrollIntoView(
                                LogListBox.Items[LogListBox.Items.Count - 1]
                            );
                        });
                    });
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                SearchProgressBar.Visibility = Visibility.Hidden;
            }
        }

        private void EmbedButton_Click(object sender, RoutedEventArgs e)
        {
            SearchProgressBar.Visibility = Visibility.Visible;
            EmbedAsync();
        }

        private void DownloadButton_Click(object sender, RoutedEventArgs e)
        {
            var download = new DownloadWindow
            {
                Owner = this
            };
            download.ShowDialog();
        }
    }
}