using System.Runtime.CompilerServices;
using MauiOnnxSample.Models;
using MauiOnnxSample.Tools;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace MauiOnnxSample.Services;

/// <summary>
/// Chat service that wraps OnnxRuntimeGenAIChatClient with:
/// - FunctionInvokingChatClient middleware for automatic tool calling
/// - RAG injection from FaqService
/// - Streaming and structured output support
/// </summary>
public class ChatService : IChatService, IDisposable
{
    private const string SystemPrompt = """
        You are a helpful AI assistant running entirely on-device using a local ONNX model.
        
        You have access to the following tools:
        - GetCurrentLocation: Gets the user's GPS coordinates
        - GetWeather: Fetches weather for given GPS coordinates
        - SwitchTheme: Changes the app theme to dark, light, or system
        
        Use tools when appropriate. When asked about weather, first get the location, then get the weather.
        Be concise but informative. Format responses for a mobile chat interface.
        
        When you have relevant FAQ context provided, use it to answer questions accurately.
        """;

    private readonly IModelService _modelService;
    private readonly FaqService _faqService;
    private readonly ChatTools _chatTools;
    private readonly ILogger<ChatService> _logger;

    private IChatClient? _chatClient;
    private bool _disposed;

    public ChatService(
        IModelService modelService,
        FaqService faqService,
        ChatTools chatTools,
        ILogger<ChatService> logger)
    {
        _modelService = modelService;
        _faqService = faqService;
        _chatTools = chatTools;
        _logger = logger;
    }

    public bool IsReady => _modelService.IsModelReady && _chatClient is not null;

    /// <summary>Lazily initializes the IChatClient when the model is ready.</summary>
    private IChatClient GetOrCreateChatClient()
    {
        if (_chatClient is not null)
            return _chatClient;

        if (!_modelService.IsModelReady || _modelService.ModelPath is null)
            throw new InvalidOperationException("Model is not ready. Call IModelService.PrepareModelAsync() first.");

        var onnxClient = new OnnxRuntimeGenAIChatClient(_modelService.ModelPath,
            new OnnxRuntimeGenAIChatClientOptions
            {
                StopSequences = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"],
                EnableCaching = false,
            });

        // Wrap with FunctionInvokingChatClient for automatic tool execution
        _chatClient = onnxClient
            .AsBuilder()
            .UseFunctionInvocation()
            .Build();

        _logger.LogInformation("Chat client initialized from model at {Path}", _modelService.ModelPath);
        return _chatClient;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> SendStreamingMessageAsync(
        IList<ChatMessage> history,
        string userMessage,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var client = GetOrCreateChatClient();

        var messages = BuildMessages(history, userMessage);
        var options = BuildChatOptions();

        _logger.LogDebug("Sending streaming message, history={Count}", messages.Count);

        await foreach (var update in client.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            var text = update.Text;
            if (!string.IsNullOrEmpty(text))
                yield return text;
        }
    }

    /// <inheritdoc/>
    public async Task<T?> SendStructuredMessageAsync<T>(
        IList<ChatMessage> history,
        string userMessage,
        CancellationToken cancellationToken = default)
    {
        var client = GetOrCreateChatClient();

        var messages = BuildMessages(history, userMessage);
        var options = BuildChatOptions();

        _logger.LogDebug("Sending structured message for type {Type}", typeof(T).Name);

        var response = await client.GetResponseAsync<T>(messages, options, cancellationToken: cancellationToken);
        return response.Result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private List<ChatMessage> BuildMessages(IList<ChatMessage> history, string userMessage)
    {
        // Retrieve relevant FAQ context for this message
        var faqContext = _faqService.BuildContextBlock(userMessage);

        var systemContent = string.IsNullOrEmpty(faqContext)
            ? SystemPrompt
            : $"{SystemPrompt}\n\n{faqContext}";

        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, systemContent)
        };

        // Add conversation history (skip any existing system messages)
        foreach (var msg in history)
        {
            if (msg.Role != ChatRole.System)
                messages.Add(msg);
        }

        messages.Add(new ChatMessage(ChatRole.User, userMessage));

        return messages;
    }

    private ChatOptions BuildChatOptions() => new()
    {
        Tools = [.. _chatTools.CreateAITools()],
        MaxOutputTokens = 2048,
        Temperature = 0.7f,
    };

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        (_chatClient as IDisposable)?.Dispose();
        _chatClient = null;
    }
}
