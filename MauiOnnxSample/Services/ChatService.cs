using System.Runtime.CompilerServices;
using System.Text.Json;
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
///
/// Phi-4-mini tool calling uses the model's native chat template via ApplyChatTemplate:
/// the system message is passed with a "tools" property containing the JSON tool definitions.
/// The model outputs tool calls using &lt;|tool_call|&gt;JSON&lt;|/tool_call|&gt; tokens.
/// Phi4ToolCallParserClient intercepts that output and converts it to FunctionCallContent.
/// </summary>
public class ChatService : IChatService, IDisposable
{
    private const string SystemPrompt = """
        You are a helpful AI assistant running entirely on-device using a local ONNX model.
        
        You have access to tools. When a user asks about location, weather, or theme, you MUST call the appropriate tool immediately — do not ask for permission.
        
        - To get location: call GetCurrentLocation with no arguments
        - To get weather: first call GetCurrentLocation, then call GetWeather with the returned coordinates
        - To change theme: call SwitchTheme with 'dark', 'light', or 'system'
        
        Always call tools directly when they are needed. Do not describe what you would do — do it.
        
        When you have relevant FAQ context provided, use it to answer questions accurately.
        """;

    private readonly IModelService _modelService;
    private readonly FaqService _faqService;
    private readonly ChatTools _chatTools;
    private readonly ILogger<ChatService> _logger;
    private readonly ILogger<Phi4ToolCallParserClient> _parserLogger;
    private readonly Lock _clientLock = new();

    private IChatClient? _chatClient;
    // Model is owned by ChatService; Tokenizer is derived from it.
    private Model? _onnxModel;
    private Tokenizer? _tokenizer;
    private bool _disposed;

    public ChatService(
        IModelService modelService,
        FaqService faqService,
        ChatTools chatTools,
        ILogger<ChatService> logger,
        ILogger<Phi4ToolCallParserClient> parserLogger)
    {
        _modelService = modelService;
        _faqService = faqService;
        _chatTools = chatTools;
        _logger = logger;
        _parserLogger = parserLogger;
    }

    // IsReady is true as soon as the model is prepared; the chat client is created lazily on first use.
    public bool IsReady => _modelService.IsModelReady;

    /// <summary>Lazily initializes the IChatClient when the model is ready (thread-safe).</summary>
    private IChatClient GetOrCreateChatClient()
    {
        if (_chatClient is not null)
            return _chatClient;

        lock (_clientLock)
        {
            // Double-checked locking
            if (_chatClient is not null)
                return _chatClient;

            if (!_modelService.IsModelReady || _modelService.ModelPath is null)
                throw new InvalidOperationException("Model is not ready. Call IModelService.PrepareModelAsync() first.");

            // Create Model and Tokenizer ourselves so we can call ApplyChatTemplate with
            // the "tools" property embedded in the system message — the only way Phi-4-mini's
            // built-in Jinja template emits the <|tool|>...<|/tool|> special tokens correctly.
            _onnxModel = new Model(_modelService.ModelPath);
            _tokenizer = new Tokenizer(_onnxModel);

            var tokenizer = _tokenizer;
            var logger = _logger;

            var onnxClient = new OnnxRuntimeGenAIChatClient(_onnxModel, ownsModel: false,
                new OnnxRuntimeGenAIChatClientOptions
                {
                    // Phi-4-mini special tokens: <|end|> terminates turns, <|endoftext|> is EOS.
                    // Also stop on role markers to prevent the model from self-continuing.
                    StopSequences = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"],
                    // Use the model's own ApplyChatTemplate with tools injected on the system message.
                    PromptFormatter = (messages, options) =>
                        FormatPromptWithTools(messages, options, tokenizer, logger),
                    EnableCaching = false,
                });

            // Pipeline: FunctionInvokingChatClient → Phi4ToolCallParserClient → OnnxRuntimeGenAIChatClient
            // Phi4ToolCallParserClient converts the model's text tool-call markers into FunctionCallContent,
            // which FunctionInvokingChatClient then detects and invokes automatically.
            _chatClient = onnxClient
                .AsBuilder()
                .Use(inner => new Phi4ToolCallParserClient(inner, _parserLogger))
                .UseFunctionInvocation()
                .Build();

            _logger.LogInformation("Chat client initialized from model at {Path}", _modelService.ModelPath);
            return _chatClient;
        }
    }

    private static int _formatCallCount = 0;

    private static string FormatPromptWithTools(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options,
        Tokenizer tokenizer,
        ILogger logger)
    {
        var callN = System.Threading.Interlocked.Increment(ref _formatCallCount);
        logger.LogInformation("FormatPromptWithTools: call #{N}, toolCount={ToolCount}",
            callN, options?.Tools?.Count ?? 0);

        try
        {
            return FormatPromptWithToolsImpl(messages, options, tokenizer, logger, callN);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "FormatPromptWithTools: EXCEPTION in call #{N}: {Msg}", callN, ex.Message);
            throw;
        }
    }

    /// <summary>
    /// Builds the prompt by calling <c>Tokenizer.ApplyChatTemplate</c> with the tools JSON
    /// embedded as a <c>"tools"</c> property on the system message.
    ///
    /// Phi-4-mini's Jinja template checks <c>message['tools']</c> on the system message and, when
    /// present, emits <c>&lt;|tool|&gt;{tools}&lt;|/tool|&gt;</c> as genuine special tokens —
    /// which is what causes the model to respond with <c>&lt;|tool_call|&gt;...&lt;|/tool_call|&gt;</c>.
    ///
    /// For the second round (after tool execution), messages include assistant FunctionCallContent
    /// and Tool FunctionResultContent. We serialize these correctly for Phi-4's template:
    ///   - Assistant with function call: role="assistant", content=JSON array
    ///   - Tool result: role="tool_response", content=result string
    /// </summary>
    private static string FormatPromptWithToolsImpl(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options,
        Tokenizer tokenizer,
        ILogger logger,
        int callN)
    {
        // Serialize AIFunction tools to the JSON schema array Phi-4-mini expects.
        string? toolsJson = null;
        if (options?.Tools is { Count: > 0 } tools)
        {
            var aiFunctions = tools.OfType<AIFunction>().ToArray();
            if (aiFunctions.Length > 0)
            {
                using var buf = new System.IO.MemoryStream();
                using var writer = new Utf8JsonWriter(buf);
                writer.WriteStartArray();
                foreach (var f in aiFunctions)
                {
                    writer.WriteStartObject();
                    writer.WriteString("type", "function");
                    writer.WritePropertyName("function");
                    writer.WriteStartObject();
                    writer.WriteString("name", f.Name);
                    writer.WriteString("description", f.Description ?? string.Empty);
                    writer.WritePropertyName("parameters");
                    f.JsonSchema.WriteTo(writer);
                    writer.WriteEndObject();
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
                writer.Flush();
                toolsJson = System.Text.Encoding.UTF8.GetString(buf.ToArray());
                logger.LogInformation("FormatPromptWithTools #{N}: {ToolCount} tools", callN, aiFunctions.Length);
            }
        }

        // Build the messages JSON array, handling all message types.
        var messageList = messages.ToList();
        logger.LogInformation("FormatPromptWithTools #{N}: {MsgCount} messages: {Roles}",
            callN, messageList.Count,
            string.Join(", ", messageList.Select(m => m.Role.Value + "(" + (m.Text?.Length > 0 ? m.Text![..Math.Min(30, m.Text.Length)] : m.Contents.FirstOrDefault()?.GetType().Name ?? "?") + ")")));

        using var msgBuf = new System.IO.MemoryStream();
        using var msgWriter = new Utf8JsonWriter(msgBuf);
        msgWriter.WriteStartArray();
        foreach (var m in messageList)
        {
            msgWriter.WriteStartObject();

            if (m.Role == ChatRole.System)
            {
                msgWriter.WriteString("role", "system");
                msgWriter.WriteString("content", m.Text ?? string.Empty);
                if (toolsJson is not null)
                    msgWriter.WriteString("tools", toolsJson);
            }
            else if (m.Role == ChatRole.Tool)
            {
                // Tool result → Phi-4 uses <|tool_response|> token.
                // ApplyChatTemplate maps role="tool_response" → <|tool_response|>content<|end|>
                msgWriter.WriteString("role", "tool_response");
                var resultText = string.Join("\n", m.Contents
                    .OfType<FunctionResultContent>()
                    .Select(r => r.Result?.ToString() ?? string.Empty));
                msgWriter.WriteString("content", resultText);
            }
            else if (m.Role == ChatRole.Assistant)
            {
                msgWriter.WriteString("role", "assistant");
                var funcCalls = m.Contents.OfType<FunctionCallContent>().ToArray();
                if (funcCalls.Length > 0)
                {
                    // Serialize the tool calls as JSON (the format the model originally produced)
                    using var callBuf = new System.IO.MemoryStream();
                    using var callWriter = new Utf8JsonWriter(callBuf);
                    callWriter.WriteStartArray();
                    foreach (var c in funcCalls)
                    {
                        callWriter.WriteStartObject();
                        callWriter.WriteString("name", c.Name);
                        callWriter.WritePropertyName("parameters");
                        callWriter.WriteStartObject();
                        if (c.Arguments is { } args)
                        {
                            foreach (var kvp in args)
                            {
                                callWriter.WritePropertyName(kvp.Key);
                                JsonSerializer.Serialize(callWriter, kvp.Value);
                            }
                        }
                        callWriter.WriteEndObject();
                        callWriter.WriteEndObject();
                    }
                    callWriter.WriteEndArray();
                    callWriter.Flush();
                    var callsJson = System.Text.Encoding.UTF8.GetString(callBuf.ToArray());
                    msgWriter.WriteString("content", callsJson);
                }
                else
                {
                    msgWriter.WriteString("content", m.Text ?? string.Empty);
                }
            }
            else
            {
                msgWriter.WriteString("role", m.Role.Value);
                msgWriter.WriteString("content", m.Text ?? string.Empty);
            }

            msgWriter.WriteEndObject();
        }
        msgWriter.WriteEndArray();
        msgWriter.Flush();
        var messagesJson = System.Text.Encoding.UTF8.GetString(msgBuf.ToArray());

        logger.LogDebug("FormatPromptWithTools #{N}: messagesJson={Json}", callN, messagesJson[..Math.Min(400, messagesJson.Length)]);

        var prompt = tokenizer.ApplyChatTemplate(null, messagesJson, null, add_generation_prompt: true);
        logger.LogInformation("FormatPromptWithTools #{N}: prompt length={Len}, snippet={Snip}",
            callN, prompt.Length, prompt[..Math.Min(400, prompt.Length)]);

        return prompt;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> SendStreamingMessageAsync(
        IList<ChatMessage> history,
        string userMessage,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var client = GetOrCreateChatClient();

        var messages = BuildMessages(history);
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

        var messages = BuildMessages(history);
        var options = BuildChatOptions();

        _logger.LogDebug("Sending structured message for type {Type}", typeof(T).Name);

        var response = await client.GetResponseAsync<T>(messages, options, cancellationToken: cancellationToken);
        return response.Result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private List<ChatMessage> BuildMessages(IList<ChatMessage> history)
    {
        // Use the last user message as the query for FAQ retrieval
        var lastUserText = history.LastOrDefault(m => m.Role == ChatRole.User)?.Text ?? string.Empty;
        var faqContext = _faqService.BuildContextBlock(lastUserText);

        var systemContent = string.IsNullOrEmpty(faqContext)
            ? SystemPrompt
            : $"{SystemPrompt}\n\n{faqContext}";

        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, systemContent)
        };

        // Add conversation history (skip any existing system messages).
        // The caller is responsible for ensuring userMessage is already the last
        // entry in history before calling this method — do NOT append it again.
        foreach (var msg in history)
        {
            if (msg.Role != ChatRole.System)
                messages.Add(msg);
        }

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
        lock (_clientLock)
        {
            if (_disposed) return;
            _disposed = true;
            (_chatClient as IDisposable)?.Dispose();
            _chatClient = null;
            _tokenizer?.Dispose();
            _tokenizer = null;
            _onnxModel?.Dispose();
            _onnxModel = null;
        }
    }
}
