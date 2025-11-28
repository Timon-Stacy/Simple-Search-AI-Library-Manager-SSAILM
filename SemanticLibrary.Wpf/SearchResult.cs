using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AILibrary
{
    internal class SearchResult
    {
        public double score { get; set; }
        public string author { get; set; }
        public string title { get; set; }
        public string category { get; set; }
        public string url { get; set; }
        public string text { get; set; }
    }
}
