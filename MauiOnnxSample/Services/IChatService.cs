using MauiOnnxSample.Models;
using Microsoft.Extensions.AI;

namespace MauiOnnxSample.Services;

/// <summary>Provides AI chat capabilities including streaming, tool calling, RAG, and structured output.</summary>
public interface IChatService
{
    /// <summary>Gets whether the underlying model is ready to accept requests.</summary>
    bool IsReady { get; }

    /// <summary>Sends a message and streams the response token by token.</summary>
    IAsyncEnumerable<string> SendStreamingMessageAsync(
        IList<ChatMessage> history,
        string userMessage,
        CancellationToken cancellationToken = default);

    /// <summary>Sends a message and returns a fully typed structured response.</summary>
    Task<T?> SendStructuredMessageAsync<T>(
        IList<ChatMessage> history,
        string userMessage,
        CancellationToken cancellationToken = default);
}
