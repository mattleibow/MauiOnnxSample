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

            // Pipeline: Phi4ToolCallParserClient → OnnxRuntimeGenAIChatClient
            // Phi4ToolCallParserClient converts the model's text tool-call markers into FunctionCallContent.
            // Tool invocation is handled manually in SendStreamingMessageAsync for reliability.
            _chatClient = onnxClient
                .AsBuilder()
                .Use(inner => new Phi4ToolCallParserClient(inner, _parserLogger))
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
        var tools = options.Tools?.OfType<AIFunction>().ToList() ?? [];

        _logger.LogInformation("SendStreamingMessage: {MsgCount} messages, {ToolCount} tools",
            messages.Count, tools.Count);

        const int maxIterations = 5;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            _logger.LogInformation("SendStreamingMessage: iteration {N}, messages={Count}",
                iteration, messages.Count);

            // Collect the full streaming response (parser buffers and converts tool calls).
            var funcCallContents = new List<FunctionCallContent>();
            var textParts = new List<string>();

            await foreach (var update in client.GetStreamingResponseAsync(messages, options, cancellationToken))
            {
                foreach (var content in update.Contents)
                {
                    if (content is FunctionCallContent fcc)
                    {
                        funcCallContents.Add(fcc);
                        _logger.LogInformation("SendStreamingMessage: tool call detected: {Name}({Args})",
                            fcc.Name,
                            fcc.Arguments is null ? "" : string.Join(", ", fcc.Arguments.Select(kv => $"{kv.Key}={kv.Value}")));
                    }
                    else if (content is TextContent tc && !string.IsNullOrEmpty(tc.Text))
                    {
                        textParts.Add(tc.Text);
                    }
                }
            }

            if (funcCallContents.Count == 0)
            {
                // No tool calls — this is the final text response.
                _logger.LogInformation("SendStreamingMessage: no tool calls, returning {Words} text chunk(s)", textParts.Count);
                foreach (var part in textParts)
                    yield return part;
                yield break;
            }

            // Add assistant's tool-call message to conversation.
            messages.Add(new ChatMessage(ChatRole.Assistant,
                [.. funcCallContents.Cast<AIContent>()]));

            // Invoke each tool and collect results.
            var toolResultContents = new List<AIContent>();
            foreach (var fcc in funcCallContents)
            {
                _logger.LogInformation("SendStreamingMessage: invoking tool '{Name}'", fcc.Name);

                var tool = tools.FirstOrDefault(t =>
                    string.Equals(t.Name, fcc.Name, StringComparison.OrdinalIgnoreCase));

                string resultText;
                if (tool is null)
                {
                    _logger.LogWarning("SendStreamingMessage: tool '{Name}' not found in options.Tools", fcc.Name);
                    resultText = $"Tool '{fcc.Name}' is not available.";
                }
                else
                {
                    try
                    {
                        var raw = await tool.InvokeAsync(
                            new AIFunctionArguments(fcc.Arguments),
                            cancellationToken);
                        resultText = raw?.ToString() ?? string.Empty;
                        _logger.LogInformation("SendStreamingMessage: tool '{Name}' returned: {Result}",
                            fcc.Name, resultText);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "SendStreamingMessage: tool '{Name}' threw: {Msg}", fcc.Name, ex.Message);
                        resultText = $"Tool error: {ex.Message}";
                    }
                }

                toolResultContents.Add(new FunctionResultContent(fcc.CallId, resultText));
            }

            // Add tool results to conversation and loop for the model's follow-up response.
            messages.Add(new ChatMessage(ChatRole.Tool, toolResultContents));
        }

        _logger.LogWarning("SendStreamingMessage: reached max iterations ({Max})", maxIterations);
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
        var tools = options.Tools?.OfType<AIFunction>().ToList() ?? [];

        _logger.LogInformation("SendStructuredMessage: type={Type}, {MsgCount} messages, {ToolCount} tools",
            typeof(T).Name, messages.Count, tools.Count);

        // Step 1: Run the tool-invocation loop to collect real data before requesting JSON.
        const int maxIterations = 5;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            var response = await client.GetResponseAsync(messages, options, cancellationToken);
            var funcCalls = response.Messages.SelectMany(m => m.Contents.OfType<FunctionCallContent>()).ToList();

            if (funcCalls.Count == 0)
            {
                _logger.LogInformation("SendStructuredMessage: no tool calls at iteration {N}, proceeding to JSON format", iteration);
                break;
            }

            _logger.LogInformation("SendStructuredMessage: iteration {N}, {Count} tool call(s)", iteration, funcCalls.Count);
            messages.Add(new ChatMessage(ChatRole.Assistant, [.. funcCalls.Cast<AIContent>()]));

            var resultContents = new List<AIContent>();
            foreach (var fcc in funcCalls)
            {
                _logger.LogInformation("SendStructuredMessage: invoking tool '{Name}'", fcc.Name);
                var tool = tools.FirstOrDefault(t => string.Equals(t.Name, fcc.Name, StringComparison.OrdinalIgnoreCase));
                string resultText;
                if (tool is null)
                {
                    _logger.LogWarning("SendStructuredMessage: tool '{Name}' not found", fcc.Name);
                    resultText = $"Tool '{fcc.Name}' is not available.";
                }
                else
                {
                    try
                    {
                        var raw = await tool.InvokeAsync(new AIFunctionArguments(fcc.Arguments), cancellationToken);
                        resultText = raw?.ToString() ?? string.Empty;
                        _logger.LogInformation("SendStructuredMessage: tool '{Name}' returned: {Result}", fcc.Name, resultText);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "SendStructuredMessage: tool '{Name}' threw: {Msg}", fcc.Name, ex.Message);
                        resultText = $"Tool error: {ex.Message}";
                    }
                }
                resultContents.Add(new FunctionResultContent(fcc.CallId, resultText));
            }
            messages.Add(new ChatMessage(ChatRole.Tool, resultContents));
        }

        // Step 2: Append a fill-in JSON template so the model outputs structured data.
        var props = typeof(T).GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
        var templateParts = props.Select(p =>
        {
            var jsonAttr = p.GetCustomAttributes(typeof(System.Text.Json.Serialization.JsonPropertyNameAttribute), false)
                            .OfType<System.Text.Json.Serialization.JsonPropertyNameAttribute>()
                            .FirstOrDefault();
            var key = jsonAttr?.Name ?? p.Name;
            var placeholder = p.PropertyType == typeof(string) ? $"\"<{key}>\"" :
                              p.PropertyType == typeof(double) || p.PropertyType == typeof(float) ? "0.0" :
                              p.PropertyType == typeof(int) ? "0" : "null";
            return $"\"{key}\": {placeholder}";
        });
        var template = "{\n  " + string.Join(",\n  ", templateParts) + "\n}";

        messages.Add(new ChatMessage(ChatRole.User,
            $"Now format the collected data as JSON. Respond with ONLY the JSON object — no prose, no markdown. Fill in real values from the tool results:\n{template}"));

        _logger.LogInformation("SendStructuredMessage: requesting JSON, template={Template}",
            template[..Math.Min(120, template.Length)]);

        // Step 3: Get JSON-formatted response and deserialize.
        var jsonResponse = await client.GetResponseAsync<T>(messages, options, cancellationToken: cancellationToken);
        _logger.LogInformation("SendStructuredMessage: result={Result}", jsonResponse.Result);
        return jsonResponse.Result;
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
