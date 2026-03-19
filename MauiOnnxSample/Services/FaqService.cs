using MauiOnnxSample.Models;
using System.Text.RegularExpressions;

namespace MauiOnnxSample.Services;

/// <summary>
/// In-memory FAQ store with TF-IDF cosine similarity for RAG retrieval.
/// Populates a set of FAQ entries about MAUI, .NET AI, and the app itself.
/// </summary>
public class FaqService
{
    private readonly List<FaqEntry> _entries = [];
    private Dictionary<string, double> _idf = [];

    public FaqService()
    {
        PopulateFaq();
        BuildIdf();
    }

    /// <summary>Retrieves the top-K most relevant FAQ entries for the given query.</summary>
    public IReadOnlyList<FaqEntry> Search(string query, int topK = 3)
    {
        if (string.IsNullOrWhiteSpace(query) || _entries.Count == 0)
            return [];

        var queryVector = BuildTermVector(Tokenize(query));

        var ranked = _entries
            .Select(e => (Entry: e, Score: CosineSimilarity(queryVector, e.TermVector)))
            .Where(x => x.Score > 0)
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .ToList();

        return ranked.Select(x => x.Entry).ToList();
    }

    /// <summary>Formats the top-K FAQ matches as a context block for injection into the system prompt.</summary>
    public string BuildContextBlock(string query, int topK = 3)
    {
        var matches = Search(query, topK);
        if (matches.Count == 0)
            return string.Empty;

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Relevant FAQ context:");
        foreach (var entry in matches)
        {
            sb.AppendLine($"Q: {entry.Question}");
            sb.AppendLine($"A: {entry.Answer}");
            sb.AppendLine();
        }
        return sb.ToString().TrimEnd();
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private void BuildIdf()
    {
        var docCount = _entries.Count;
        var dfMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var entry in _entries)
        {
            var terms = Tokenize($"{entry.Question} {entry.Answer}");
            foreach (var term in terms.Distinct())
                dfMap[term] = dfMap.TryGetValue(term, out var v) ? v + 1 : 1;
        }

        _idf = dfMap.ToDictionary(
            kv => kv.Key,
            kv => Math.Log((docCount + 1.0) / (kv.Value + 1.0)) + 1.0,
            StringComparer.OrdinalIgnoreCase);

        // Pre-compute TF-IDF vectors for all entries
        foreach (var entry in _entries)
        {
            var terms = Tokenize($"{entry.Question} {entry.Answer}");
            entry.TermVector = BuildTermVector(terms);
        }
    }

    private Dictionary<string, double> BuildTermVector(IEnumerable<string> terms)
    {
        var tf = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        int total = 0;
        foreach (var term in terms)
        {
            tf[term] = tf.TryGetValue(term, out var v) ? v + 1 : 1;
            total++;
        }

        if (total == 0) return [];

        return tf.ToDictionary(
            kv => kv.Key,
            kv =>
            {
                double termFreq = (double)kv.Value / total;
                double idfVal = _idf.TryGetValue(kv.Key, out var i) ? i : 1.0;
                return termFreq * idfVal;
            },
            StringComparer.OrdinalIgnoreCase);
    }

    private static double CosineSimilarity(Dictionary<string, double> a, Dictionary<string, double> b)
    {
        double dot = 0, normA = 0, normB = 0;
        foreach (var (term, valA) in a)
        {
            normA += valA * valA;
            if (b.TryGetValue(term, out var valB))
                dot += valA * valB;
        }
        foreach (var valB in b.Values)
            normB += valB * valB;

        if (normA == 0 || normB == 0) return 0;
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    private static IEnumerable<string> Tokenize(string text) =>
        Regex.Split(text.ToLowerInvariant(), @"[^a-z0-9]+")
             .Where(t => t.Length > 2);

    // ── FAQ data ─────────────────────────────────────────────────────────────

    private void PopulateFaq()
    {
        AddFaq("What is this app?",
            "MauiOnnxSample is an AI chat application built with .NET MAUI that runs a local AI model (Phi-3.5-mini) entirely on your device using ONNX Runtime GenAI. It supports tool calling, RAG, and structured outputs.",
            "App");

        AddFaq("What AI model does this app use?",
            "The app uses Microsoft's Phi-3.5-mini-instruct model in INT4 quantized ONNX format. It runs completely on-device with no internet connection required for the AI inference itself.",
            "App");

        AddFaq("Does this app send my data to the cloud?",
            "No. The AI model runs entirely on your device. Your conversations stay private. The weather tool makes requests to the free Open-Meteo API, but no personal data is sent.",
            "Privacy");

        AddFaq("What is MAUI?",
            ".NET MAUI (Multi-platform App UI) is a cross-platform framework for building native mobile and desktop apps with C# and XAML. It lets you target Android, iOS, macOS, and Windows from a single codebase.",
            "MAUI");

        AddFaq("What platforms does MAUI support?",
            ".NET MAUI supports Android, iOS, macOS (via Mac Catalyst), and Windows. This app runs natively on all four platforms.",
            "MAUI");

        AddFaq("What is ONNX?",
            "ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. It allows models trained in frameworks like PyTorch or TensorFlow to run on various runtimes, including ONNX Runtime.",
            "AI");

        AddFaq("What is ONNX Runtime GenAI?",
            "ONNX Runtime GenAI is a library from Microsoft that enables running generative AI models (like LLMs) using ONNX Runtime. It handles tokenization, the generation loop, KV cache management, and search strategies.",
            "AI");

        AddFaq("What is Microsoft.Extensions.AI?",
            "Microsoft.Extensions.AI provides .NET abstractions for AI services, including IChatClient and IEmbeddingGenerator interfaces. It enables writing AI-powered code that works with different model providers through a unified API.",
            "AI");

        AddFaq("What is RAG?",
            "RAG (Retrieval-Augmented Generation) is a technique where relevant documents or knowledge base entries are retrieved and injected into the AI's context before generating a response. This allows the AI to answer questions based on specific knowledge without retraining.",
            "AI");

        AddFaq("What is tool calling?",
            "Tool calling (also known as function calling) is a capability where an AI model can request the execution of external functions. The model outputs a structured request indicating which tool to call and with what arguments; your code runs the tool and returns results to the model.",
            "AI");

        AddFaq("How do I get the weather?",
            "Ask the AI about the current weather. It will use the GPS tool to get your location and then fetch weather data from the Open-Meteo API (free, no API key needed). Example: 'What's the weather like right now?'",
            "Features");

        AddFaq("How do I switch the app theme?",
            "Ask the AI to switch the theme. Examples: 'Switch to dark mode', 'Use light theme', 'Use system theme'. The AI will invoke the theme-switching tool on your behalf.",
            "Features");

        AddFaq("What is streaming in AI chat?",
            "Streaming means the AI's response appears token by token as it's generated, rather than waiting for the complete response. This provides a more responsive experience and lets you start reading immediately.",
            "AI");

        AddFaq("What is structured output?",
            "Structured output is when the AI returns a response in a specific JSON format matching a predefined schema (like WeatherInfo with temperature, humidity, etc.). This makes it easy to programmatically process AI responses.",
            "AI");

        AddFaq("How large is the AI model?",
            "The Phi-3.5-mini-instruct model in INT4 quantized format is approximately 2.3 GB. It needs to be downloaded separately using the provided scripts and placed in the model assets directory.",
            "App");

        AddFaq("What is .NET 10?",
            ".NET 10 is the latest LTS (Long-Term Support) release of Microsoft's cross-platform development framework. It includes C# 14, performance improvements, and new APIs including better AI integration through Microsoft.Extensions.AI.",
            "Development");

        AddFaq("What is C# 14?",
            "C# 14 is the version of the C# programming language shipped with .NET 10. It introduces new features like field-backed properties, unmanaged generic constraints, and more collection expression improvements.",
            "Development");

        AddFaq("How does GPS work in this app?",
            "The app uses .NET MAUI's Geolocation API to request your device's current GPS coordinates. This requires location permission to be granted. The coordinates are only used for fetching weather data.",
            "Features");

        AddFaq("What is a vector store?",
            "A vector store is a database that stores data as high-dimensional vectors (embeddings). It enables semantic similarity search — finding documents that are conceptually similar to a query, not just keyword matches. This app uses an in-memory TF-IDF vector store for FAQ retrieval.",
            "AI");

        AddFaq("What is Open-Meteo?",
            "Open-Meteo is a free, open-source weather API that provides weather forecasts and historical data. It requires no API key and supports global coverage. This app uses it to fetch current weather conditions.",
            "Features");
    }

    private void AddFaq(string question, string answer, string category) =>
        _entries.Add(new FaqEntry(question, answer, category));
}
