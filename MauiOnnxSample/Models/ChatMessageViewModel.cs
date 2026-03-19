namespace MauiOnnxSample.Models;

/// <summary>Represents a single message in the chat conversation.</summary>
public class ChatMessageViewModel
{
    public ChatMessageViewModel(bool isUser, string text)
    {
        IsUser = isUser;
        Text = text;
        Timestamp = DateTime.Now;
        IsStreaming = false;
    }

    public bool IsUser { get; }
    public string Text { get; set; }
    public DateTime Timestamp { get; }
    public bool IsStreaming { get; set; }

    /// <summary>True for assistant messages, false for user.</summary>
    public bool IsAssistant => !IsUser;

    public string AuthorLabel => IsUser ? "You" : "AI";

    public override string ToString() => $"[{AuthorLabel}] {Text}";
}
