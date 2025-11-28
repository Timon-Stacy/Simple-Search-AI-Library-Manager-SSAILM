using System;
using System.Collections.Generic;
using System.Text;

namespace SemanticLibrary.Wpf
{
    class Settings
    {
        public string PythonPath { get; set; }
        public string DatabasePath { get; set; }
        public string IndexPath { get; set; }
        public string Model { get; set; }
        public int TopK { get; set; }
        public double MinCosine { get; set; }
        public int FetchK { get; set; }
    }
}