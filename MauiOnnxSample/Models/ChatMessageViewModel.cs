using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace MauiOnnxSample.Models;

/// <summary>Represents a single message in the chat conversation.</summary>
public class ChatMessageViewModel : INotifyPropertyChanged
{
    private string _text;
    private bool _isStreaming;

    public ChatMessageViewModel(bool isUser, string text)
    {
        IsUser = isUser;
        _text = text;
        Timestamp = DateTime.Now;
        _isStreaming = false;
    }

    public bool IsUser { get; }
    public DateTime Timestamp { get; }

    public string Text
    {
        get => _text;
        set => SetProperty(ref _text, value);
    }

    public bool IsStreaming
    {
        get => _isStreaming;
        set => SetProperty(ref _isStreaming, value);
    }

    /// <summary>True for assistant messages, false for user.</summary>
    public bool IsAssistant => !IsUser;

    public string AuthorLabel => IsUser ? "You" : "AI";

    public override string ToString() => $"[{AuthorLabel}] {Text}";

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        return true;
    }
}
