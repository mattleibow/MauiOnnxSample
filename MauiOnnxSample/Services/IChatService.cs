using MauiOnnxSample.Models;
using Microsoft.Extensions.AI;

namespace MauiOnnxSample.Services;

/// <summary>Provides AI chat capabilities including streaming, tool calling, RAG, and structured output.</summary>
public interface IChatService
{
    /// <summary>Gets whether the underlying model is ready to accept requests.</summary>
    bool IsReady { get; }

    /// <summary>
    /// Sends a message and streams the response token by token.
    /// The <paramref name="history"/> must already contain the new user message as its last entry.
    /// <paramref name="userMessage"/> is kept for API symmetry but the canonical source of truth is history.
    /// </summary>
    IAsyncEnumerable<string> SendStreamingMessageAsync(
        IList<ChatMessage> history,
        string userMessage,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Sends a message and returns a fully typed structured response.
    /// The <paramref name="history"/> must already contain the new user message as its last entry.
    /// </summary>
    Task<T?> SendStructuredMessageAsync<T>(
        IList<ChatMessage> history,
        string userMessage,
        CancellationToken cancellationToken = default);
}
