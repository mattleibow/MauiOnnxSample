using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;

namespace MauiOnnxSample.Services;

/// <summary>
/// Formats messages for Phi-4-mini using its special-token prompt format with embedded tool definitions.
///
/// Phi-4-mini chat template (from tokenizer_config.json):
///   System with tools:  &lt;|system|&gt;TEXT&lt;|tool|&gt;TOOLS_JSON&lt;|/tool|&gt;&lt;|end|&gt;
///   System without:     &lt;|system|&gt;TEXT&lt;|end|&gt;
///   User:               &lt;|user|&gt;TEXT&lt;|end|&gt;
///   Assistant text:     &lt;|assistant|&gt;TEXT&lt;|end|&gt;
///   Assistant toolcall: &lt;|assistant|&gt;&lt;|tool_call|&gt;JSON&lt;|/tool_call|&gt;&lt;|end|&gt;
///   Tool response:      &lt;|tool_response|&gt;TEXT&lt;|end|&gt;
///   Generation prompt:  &lt;|assistant|&gt;
/// </summary>
public sealed class Phi4PromptFormatter
{
    private readonly ILogger<Phi4PromptFormatter> _logger;

    public Phi4PromptFormatter(ILogger<Phi4PromptFormatter> logger)
    {
        _logger = logger;
    }

    public string Format(IEnumerable<ChatMessage> messages, ChatOptions? options)
    {
        var sb = new StringBuilder();
        string? toolsJson = BuildToolsJson(options?.Tools);

        _logger.LogInformation(
            "Phi4Formatter: building prompt. Tools={ToolCount}, ToolsJson={ToolsSnippet}",
            options?.Tools?.Count ?? 0,
            toolsJson is not null ? toolsJson.Substring(0, Math.Min(200, toolsJson.Length)) : "(none)");

        foreach (var message in messages)
        {
            if (message.Role == ChatRole.System)
            {
                sb.Append("<|system|>");
                sb.Append(message.Text ?? string.Empty);
                if (toolsJson is not null)
                {
                    sb.Append("<|tool|>");
                    sb.Append(toolsJson);
                    sb.Append("<|/tool|>");
                }
                sb.Append("<|end|>");
            }
            else if (message.Role == ChatRole.User)
            {
                sb.Append("<|user|>");
                sb.Append(message.Text ?? string.Empty);
                sb.Append("<|end|>");
            }
            else if (message.Role == ChatRole.Assistant)
            {
                sb.Append("<|assistant|>");
                foreach (var content in message.Contents)
                {
                    if (content is TextContent tc)
                        sb.Append(tc.Text);
                    else if (content is FunctionCallContent fcc)
                    {
                        sb.Append("<|tool_call|>");
                        sb.Append(SerializeFunctionCall(fcc));
                        sb.Append("<|/tool_call|>");
                    }
                }
                sb.Append("<|end|>");
            }
            else if (message.Role == ChatRole.Tool)
            {
                foreach (var content in message.Contents)
                {
                    if (content is FunctionResultContent frc)
                    {
                        sb.Append("<|tool_response|>");
                        sb.Append(frc.Result?.ToString() ?? string.Empty);
                        sb.Append("<|end|>");
                    }
                }
            }
        }

        sb.Append("<|assistant|>");
        var result = sb.ToString();

        _logger.LogInformation(
            "Phi4Formatter: prompt built. Length={Length}, Preview={Preview}",
            result.Length,
            result.Substring(0, Math.Min(500, result.Length)).Replace("\n", "\\n"));

        // Write debug file to app sandbox for external inspection
        try
        {
            var debugPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "phi4-prompt-debug.txt");
            File.WriteAllText(debugPath, result);
            _logger.LogDebug("Phi4Formatter: debug prompt written to {Path}", debugPath);
        }
        catch (Exception ex)
        {
            _logger.LogDebug("Phi4Formatter: could not write debug file: {Msg}", ex.Message);
        }

        return result;
    }

    private static string? BuildToolsJson(IList<AITool>? tools)
    {
        if (tools is null || tools.Count == 0) return null;

        var toolDefs = new List<object>();
        foreach (var tool in tools)
        {
            if (tool is AIFunctionDeclaration funcDecl)
            {
                JsonNode? paramsNode = null;
                try
                {
                    paramsNode = JsonNode.Parse(funcDecl.JsonSchema.GetRawText());
                }
                catch { /* fall back to empty object schema */ }

                toolDefs.Add(new
                {
                    type = "function",
                    function = new
                    {
                        name = funcDecl.Name,
                        description = funcDecl.Description,
                        parameters = paramsNode ?? (object)new { type = "object", properties = new { } }
                    }
                });
            }
        }

        return toolDefs.Count > 0 ? JsonSerializer.Serialize(toolDefs) : null;
    }

    private static string SerializeFunctionCall(FunctionCallContent fcc) =>
        JsonSerializer.Serialize(new
        {
            name = fcc.Name,
            arguments = fcc.Arguments ?? (IDictionary<string, object?>)new Dictionary<string, object?>()
        });
}

/// <summary>
/// Delegating chat client that sits between <see cref="FunctionInvokingChatClient"/> and
/// <see cref="Microsoft.ML.OnnxRuntimeGenAI.OnnxRuntimeGenAIChatClient"/>.
///
/// Phi-4-mini may emit tool calls in two formats:
///   Format A: &lt;|tool_call|&gt;{"name":"...","arguments":{...}}&lt;|/tool_call|&gt;
///   Format B: [{"name":"...","parameters":{...}}]  (JSON array, no wrapper tokens)
///
/// This client buffers the full streamed response, determines if it is a tool call,
/// and — when it is — converts it to <see cref="FunctionCallContent"/> so that
/// <see cref="FunctionInvokingChatClient"/> can detect and invoke the tools automatically.
///
/// For ordinary (non-tool-call) responses, text is passed through unchanged.
/// </summary>
public sealed class Phi4ToolCallParserClient : DelegatingChatClient
{
    private const string ToolCallStart = "<|tool_call|>";
    private const string ToolCallEnd = "<|/tool_call|>";

    private readonly ILogger<Phi4ToolCallParserClient> _logger;
    private int _callCount;

    public Phi4ToolCallParserClient(IChatClient innerClient, ILogger<Phi4ToolCallParserClient> logger)
        : base(innerClient)
    {
        _logger = logger;
    }

    public override async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var response = await base.GetResponseAsync(messages, options, cancellationToken);
        var text = response.Messages.LastOrDefault()?.Text;
        _logger.LogInformation("Phi4Parser (non-streaming): raw output={Raw}", text);
        return TryConvertToFunctionCalls(response, text) ?? response;
    }

    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var callN = System.Threading.Interlocked.Increment(ref _callCount);
        _logger.LogInformation("Phi4Parser: GetStreamingResponseAsync call #{N}, msgCount={Count}",
            callN, messages.Count());

        // Always buffer the full response before deciding: Phi-4-mini emits tool calls as
        // a plain JSON array (no <|tool_call|> wrapper), so we can only know it's a tool
        // call after we have the complete output.
        var buffer = new List<ChatResponseUpdate>();
        var textBuffer = new StringBuilder();

        await foreach (var update in base.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            if (!string.IsNullOrEmpty(update.Text))
            {
                textBuffer.Append(update.Text);
                _logger.LogDebug("Phi4Parser chunk: '{Text}'", update.Text);
            }
            buffer.Add(update);
        }

        var fullText = textBuffer.ToString();
        _logger.LogInformation("Phi4Parser: stream ended. fullText='{Text}'",
            fullText.Substring(0, Math.Min(300, fullText.Length)));

        var funcCalls = TryParseToolCalls(fullText);
        if (funcCalls is { Count: > 0 })
        {
            _logger.LogInformation("Phi4Parser: emitting {Count} FunctionCallContent(s): {Names}",
                funcCalls.Count,
                string.Join(", ", funcCalls.Select(f => f.Name)));

            var last = buffer.LastOrDefault();
            yield return new ChatResponseUpdate
            {
                Role = ChatRole.Assistant,
                Contents = [.. funcCalls.Cast<AIContent>()],
                FinishReason = ChatFinishReason.ToolCalls,
                ConversationId = last?.ConversationId,
                CreatedAt = last?.CreatedAt ?? DateTimeOffset.UtcNow,
                ResponseId = last?.ResponseId,
            };
            _logger.LogInformation("Phi4Parser: FunctionCallContent yielded, call #{N} complete", callN);
        }
        else
        {
            foreach (var buffered in buffer)
                yield return buffered;
        }
    }

    private ChatResponse? TryConvertToFunctionCalls(ChatResponse response, string? text)
    {
        if (string.IsNullOrEmpty(text)) return null;

        var funcCalls = TryParseToolCalls(text);
        if (funcCalls is null or { Count: 0 }) return null;

        _logger.LogInformation("Phi4Parser: tool call(s) detected (non-streaming): {Names}",
            string.Join(", ", funcCalls.Select(f => f.Name)));
        return new ChatResponse([new ChatMessage(ChatRole.Assistant, [.. funcCalls.Cast<AIContent>()])])
        {
            FinishReason = ChatFinishReason.ToolCalls,
        };
    }

    /// <summary>
    /// Parses the raw model output text into a list of <see cref="FunctionCallContent"/>.
    /// Returns <c>null</c> or empty when the text is not a tool call.
    /// </summary>
    private List<FunctionCallContent>? TryParseToolCalls(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;

        var trimmed = text.Trim();

        // Format A: <|tool_call|>{"name":"...","arguments":{...}}<|/tool_call|>
        if (trimmed.StartsWith(ToolCallStart, StringComparison.Ordinal))
        {
            var startIdx = ToolCallStart.Length;
            var endIdx = trimmed.IndexOf(ToolCallEnd, StringComparison.Ordinal);
            var json = endIdx > startIdx
                ? trimmed.Substring(startIdx, endIdx - startIdx).Trim()
                : trimmed.Substring(startIdx).Trim();

            // The content after <|tool_call|> may be a JSON array (multiple calls) or single object.
            if (json.StartsWith('[') || json.StartsWith('{'))
                return TryParseToolCallArray(json);

            var call = TryParseOneToolCall(json);
            return call is not null ? [call] : null;
        }

        // Format B: [{"name":"...","parameters":{...}},...] or {"name":"...","parameters":{...}}
        if (trimmed.StartsWith('[') || trimmed.StartsWith('{'))
        {
            return TryParseToolCallArray(trimmed);
        }

        return null;
    }

    /// <summary>
    /// Tries to parse text as a JSON array (or object) of tool call descriptors.
    /// Handles extra trailing brackets emitted by the model (e.g., <c>]]</c>).
    /// </summary>
    private List<FunctionCallContent>? TryParseToolCallArray(string text)
    {
        // Strip trailing extraneous characters that the model may append.
        var sanitized = text.TrimEnd();
        while (sanitized.EndsWith("]]") || sanitized.EndsWith("] ]"))
            sanitized = sanitized.Substring(0, sanitized.LastIndexOf(']')).TrimEnd();

        try
        {
            using var doc = JsonDocument.Parse(sanitized);
            var calls = new List<FunctionCallContent>();

            if (doc.RootElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var element in doc.RootElement.EnumerateArray())
                {
                    var call = TryParseOneToolCallElement(element);
                    if (call is not null)
                        calls.Add(call);
                }
            }
            else if (doc.RootElement.ValueKind == JsonValueKind.Object)
            {
                var call = TryParseOneToolCallElement(doc.RootElement);
                if (call is not null)
                    calls.Add(call);
            }

            return calls.Count > 0 ? calls : null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Phi4Parser: JSON parse failed for: {Text}",
                sanitized.Substring(0, Math.Min(200, sanitized.Length)));
            return null;
        }
    }

    private FunctionCallContent? TryParseOneToolCall(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;

        // Find the first '{' in case there's a prefix
        if (!json.StartsWith('{'))
        {
            var idx = json.IndexOf('{');
            if (idx < 0) return null;
            json = json.Substring(idx);
        }

        try
        {
            using var doc = JsonDocument.Parse(json);
            return TryParseOneToolCallElement(doc.RootElement);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Phi4Parser: could not parse tool call JSON: {Json}",
                json.Substring(0, Math.Min(200, json.Length)));
            return null;
        }
    }

    private static FunctionCallContent? TryParseOneToolCallElement(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object) return null;

        if (!element.TryGetProperty("name", out var nameProp)) return null;
        var name = nameProp.GetString();
        if (string.IsNullOrEmpty(name)) return null;

        var arguments = new Dictionary<string, object?>();

        // Accept "arguments" (canonical) or "parameters" (Phi-4 JSON array format)
        JsonElement argsEl = default;
        bool found = element.TryGetProperty("arguments", out argsEl) ||
                     element.TryGetProperty("parameters", out argsEl);

        if (found && argsEl.ValueKind == JsonValueKind.Object)
        {
            foreach (var prop in argsEl.EnumerateObject())
            {
                arguments[prop.Name] = prop.Value.ValueKind switch
                {
                    JsonValueKind.String => prop.Value.GetString(),
                    JsonValueKind.Number =>
                        prop.Value.TryGetInt32(out var i) ? (object?)i : prop.Value.GetDouble(),
                    JsonValueKind.True => (object?)true,
                    JsonValueKind.False => (object?)false,
                    JsonValueKind.Null => null,
                    _ => prop.Value.GetRawText()
                };
            }
        }

        return new FunctionCallContent(
            callId: Guid.NewGuid().ToString("N"),
            name: name,
            arguments: arguments);
    }
}
