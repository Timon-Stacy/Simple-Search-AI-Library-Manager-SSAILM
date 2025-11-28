using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using System.Windows.Forms;

namespace AILibrary
{
    internal static class EmbedUtil
    {
        public static void Embed(string pythonPath, string scriptPath, string DB, string FAISS, Action<string>? onLog = null)
        {
            var psi = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments =
                    $"\"{scriptPath}\" " +
                    $"--db \"{DB}\" " +
                    $"--index \"{FAISS}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using (var process = new Process { StartInfo = psi, EnableRaisingEvents = true })
            {
                process.OutputDataReceived += (s, e) =>
                {
                    if (e.Data != null)
                        onLog?.Invoke(e.Data);
                };

                process.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrWhiteSpace(e.Data))
                        onLog?.Invoke("[ERR] " + e.Data);
                };

                try
                {

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    process.WaitForExit();

                    if (process.ExitCode != 0)
                        onLog?.Invoke($"[ERR] Python exited with code {process.ExitCode}");
                }

                catch (Exception ex)
                {
                    onLog?.Invoke("[EXCEPTION] " + ex.ToString());
                    throw;
                }
                return;
            }
        }

    }
}