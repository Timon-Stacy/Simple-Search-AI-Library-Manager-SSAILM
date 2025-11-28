using Newtonsoft.Json;
using SemanticLibrary.Wpf;
using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace AILibrary
{
    internal static class DownloadUtil
    {
        public static async Task DownloadAsync(
            ObservableCollection<DownloadClass> items,
            string pythonExePath,
            string scriptPath,
            string databasePath,
            Action<string> logLine,
            string apiKey = null)
        {
            string json = JsonConvert.SerializeObject(items);

            // Build arguments with --db parameter
            string arguments = $"-u \"{scriptPath}\" --db \"{databasePath}\"";

            // Add API key if provided
            if (!string.IsNullOrWhiteSpace(apiKey))
            {
                arguments += $" --api-key \"{apiKey}\"";
            }

            var psi = new ProcessStartInfo
            {
                FileName = pythonExePath,
                Arguments = arguments,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = new Process
            {
                StartInfo = psi,
                EnableRaisingEvents = true
            };

            // Stream stdout
            process.OutputDataReceived += (_, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    logLine(e.Data);
            };

            // Stream stderr as error lines
            process.ErrorDataReceived += (_, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                    logLine("[ERR] " + e.Data);
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            // Send JSON to stdin
            await using (var sw = process.StandardInput)
            {
                await sw.WriteAsync(json);
            }

            await Task.Run(() => process.WaitForExit());

            if (process.ExitCode != 0)
                throw new Exception($"download.py exited with code {process.ExitCode}");
        }
    }
}