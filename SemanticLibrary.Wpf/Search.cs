using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using Newtonsoft.Json;
//using System.Windows.Forms;

namespace AILibrary
{
    internal static class Search
    {
        public static List<SearchResult> PythonSearch(
            string pythonPath,
            string scriptPath,
            string query,
            int k,
            double minCos,
            int fetchK,
            string model,
            string db,
            string index,
            bool semanticSearch)
        {
            try
            {
                Debug.WriteLine($"Search params: k={k}, minCos={minCos}, fetchK={fetchK}, model={model}");

                // naive quoting; good enough for normal use
                string quotedQuery = $"\"{query}\"";
                string quotedModel = $"\"{model}\"";
                string quotedDb = $"\"{db}\"";
                string quotedIndex = $"\"{index}\"";
                string minCosStr = minCos.ToString(CultureInfo.InvariantCulture);

                var psi = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments =
                        $"\"{scriptPath}\" " +
                        $"--q {quotedQuery} " +
                        $"--k {k} " +
                        $"--min_score {minCos} " +
                        $"--fetch_k {fetchK} " +
                        $"--model_name {quotedModel} " +
                        $"--db {quotedDb} " +
                        $"--index {quotedIndex}" +
                        $"--literal {semanticSearch}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    StandardOutputEncoding = Encoding.UTF8
                };

                try
                {
                    using (var process = Process.Start(psi))
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        string error = process.StandardError.ReadToEnd();
                        process.WaitForExit();

                        // DEBUG: Always show stderr to see debug messages
                        if (!string.IsNullOrWhiteSpace(error))
                        {
                            Debug.WriteLine("Python stderr:\n" + error);
                        }

                        if (process.ExitCode != 0)
                        {
                            throw new Exception("Python error:\n" + error);
                        }

                        if (string.IsNullOrWhiteSpace(output))
                            return new List<SearchResult>();

                        var results = JsonConvert.DeserializeObject<List<SearchResult>>(output);
                        return results ?? new List<SearchResult>();
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine("Failed to run Python process: " + ex);
                    throw;
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine("Failed to run Python: " + ex);
                return new List<SearchResult>();
            }
        }
    }
}