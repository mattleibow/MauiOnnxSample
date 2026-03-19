namespace MauiOnnxSample.Models;

/// <summary>A FAQ entry for the in-memory vector store used for RAG.</summary>
public class FaqEntry
{
    public FaqEntry(string question, string answer, string category = "General")
    {
        Question = question;
        Answer = answer;
        Category = category;
        Id = Guid.NewGuid().ToString("N");
    }

    public string Id { get; }
    public string Question { get; }
    public string Answer { get; }
    public string Category { get; }

    /// <summary>TF-IDF-style term frequency vector, populated by FaqService.</summary>
    public Dictionary<string, double> TermVector { get; set; } = [];

    public override string ToString() => $"Q: {Question}\nA: {Answer}";
}
