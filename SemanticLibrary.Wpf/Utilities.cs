using Microsoft.Data.Sqlite;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.IO;

//using System.Windows.Forms;

namespace SemanticLibrary.Wpf
{
    internal static class Utilities
    {
        public static DataTable LoadData(string path)
        {
            try
            {
                using (var conn = new SqliteConnection($"Data Source={path}"))
                using (var cmd = new SqliteCommand("SELECT id, author, title, category, source_url FROM books;", conn))
                {
                    conn.Open();

                    using (var reader = cmd.ExecuteReader())
                    {
                        var table = new DataTable();

                        // 1) Build columns with NO constraints
                        for (int i = 0; i < reader.FieldCount; i++)
                        {
                            // use object as type to be extra safe
                            table.Columns.Add(reader.GetName(i), typeof(object));
                        }

                        // 2) Copy all rows manually
                        while (reader.Read())
                        {
                            var row = table.NewRow();
                            for (int i = 0; i < reader.FieldCount; i++)
                            {
                                row[i] = reader.IsDBNull(i) ? DBNull.Value : reader.GetValue(i);
                            }
                            table.Rows.Add(row);
                        }

                        return table;
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error loading data: " + ex.Message);
                return null;
            }
        }

        public static void SaveConfig(string pythonPath, string databasePath, string indexPath, string model, int topK, double minCosine, int fetchK)
        {
            var settings = new Settings
            {
                PythonPath = pythonPath,
                DatabasePath = databasePath,
                IndexPath = indexPath,
                Model = model,
                TopK = topK,
                MinCosine = minCosine,
                FetchK = fetchK
            };

            string json = JsonConvert.SerializeObject(settings, Formatting.Indented);

            File.WriteAllText("config.json", json);
        }

        public static void LoadConfig(out string pythonPath, out string databasePath, out string indexPath, out string model, out int topK, out double minCosine, out int fetchK)
        {
            pythonPath = "python"; // Default to system python
            databasePath = "";
            indexPath = "";
            model = "sentence-transformers/all-MiniLM-L6-v2";
            topK = 5;
            minCosine = -1.0;
            fetchK = 200;

            if (File.Exists("config.json"))
            {
                string json = File.ReadAllText("config.json");
                var settings = JsonConvert.DeserializeObject<Settings>(json);
                if (settings != null)
                {
                    pythonPath = settings.PythonPath ?? "python";
                    databasePath = settings.DatabasePath;
                    indexPath = settings.IndexPath;
                    model = settings.Model;
                    topK = settings.TopK;
                    minCosine = settings.MinCosine;
                    fetchK = settings.FetchK;
                }
            }
        }

        public static string GetScriptPath(string scriptName)
        {
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            return Path.Combine(exeDir, scriptName);
        }
    }
}