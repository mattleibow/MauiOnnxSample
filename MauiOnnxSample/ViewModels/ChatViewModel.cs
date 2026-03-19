using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using MauiOnnxSample.Models;
using MauiOnnxSample.Services;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;

namespace MauiOnnxSample.ViewModels;

/// <summary>
/// ViewModel for the chat page. Manages conversation history, model loading,
/// streaming UI updates, and coordinates with IChatService.
/// </summary>
public class ChatViewModel : INotifyPropertyChanged
{
    private readonly IChatService _chatService;
    private readonly IModelService _modelService;
    private readonly ILogger<ChatViewModel> _logger;

    private readonly List<ChatMessage> _chatHistory = [];
    private string _inputText = string.Empty;
    private bool _isBusy;
    private bool _isModelLoading = true;
    private string _statusMessage = "Initializing model...";
    private CancellationTokenSource? _streamCts;

    public ChatViewModel(
        IChatService chatService,
        IModelService modelService,
        ILogger<ChatViewModel> logger)
    {
        _chatService = chatService;
        _modelService = modelService;
        _logger = logger;

        Messages = [];
        SendCommand = new Command(async () => await SendAsync(), CanSend);
        StopCommand = new Command(StopGeneration);
        PrepareModelCommand = new Command(async () => await PrepareModelAsync());
    }

    public ObservableCollection<ChatMessageViewModel> Messages { get; }

    public Command SendCommand { get; }
    public Command StopCommand { get; }
    public Command PrepareModelCommand { get; }

    public string InputText
    {
        get => _inputText;
        set { SetProperty(ref _inputText, value); SendCommand.ChangeCanExecute(); }
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set { SetProperty(ref _isBusy, value); SendCommand.ChangeCanExecute(); StopCommand.ChangeCanExecute(); }
    }

    public bool IsModelLoading
    {
        get => _isModelLoading;
        private set { SetProperty(ref _isModelLoading, value); SendCommand.ChangeCanExecute(); }
    }

    public string StatusMessage
    {
        get => _statusMessage;
        private set => SetProperty(ref _statusMessage, value);
    }

    public bool IsModelReady => _chatService.IsReady;

    public bool CanSend() => !IsBusy && !IsModelLoading && !string.IsNullOrWhiteSpace(InputText) && _chatService.IsReady;

    // ── Model preparation ─────────────────────────────────────────────────────

    public async Task PrepareModelAsync()
    {
        IsModelLoading = true;
        StatusMessage = "Preparing model...";

        var progress = new Progress<string>(msg =>
        {
            MainThread.BeginInvokeOnMainThread(() => StatusMessage = msg);
        });

        bool ready = await _modelService.PrepareModelAsync(progress);

        MainThread.BeginInvokeOnMainThread(() =>
        {
            IsModelLoading = false;
            OnPropertyChanged(nameof(IsModelReady));

            if (ready)
            {
                StatusMessage = "Model ready. Start chatting!";
                AddSystemInfo("AI model loaded. I'm ready to chat! I can help with questions, get weather, switch themes, and more. Try asking: \"What's the weather like?\" or \"Switch to dark mode\".");
            }
            else
            {
                StatusMessage = "Model not found. Run download-model script.";
                AddSystemInfo(
                    "⚠️ Model not found.\n\n" +
                    "To use this app, download the Phi-3.5-mini ONNX model:\n\n" +
                    "  macOS/Linux: ./scripts/download-model.sh\n" +
                    "  Windows: .\\scripts\\download-model.ps1\n\n" +
                    "The model is ~2.3 GB and will be extracted on first launch.");
            }

            SendCommand.ChangeCanExecute();
        });
    }

    // ── Chat ──────────────────────────────────────────────────────────────────

    private async Task SendAsync()
    {
        var userText = InputText.Trim();
        if (string.IsNullOrWhiteSpace(userText)) return;

        InputText = string.Empty;
        IsBusy = true;
        _streamCts = new CancellationTokenSource();

        // Check if this is a structured output request (weather query)
        bool isWeatherQuery = userText.Contains("weather", StringComparison.OrdinalIgnoreCase)
            && (userText.Contains("json", StringComparison.OrdinalIgnoreCase)
                || userText.Contains("structured", StringComparison.OrdinalIgnoreCase));

        AddUserMessage(userText);
        _chatHistory.Add(new ChatMessage(ChatRole.User, userText));

        try
        {
            if (isWeatherQuery)
            {
                await SendStructuredWeatherAsync(userText, _streamCts.Token);
            }
            else
            {
                await SendStreamingAsync(userText, _streamCts.Token);
            }
        }
        catch (OperationCanceledException)
        {
            if (Messages.LastOrDefault() is { IsAssistant: true } lastMsg)
            {
                if (string.IsNullOrWhiteSpace(lastMsg.Text))
                    Messages.Remove(lastMsg);
                else
                    lastMsg.IsStreaming = false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during chat");
            AddSystemInfo($"Error: {ex.Message}");
        }
        finally
        {
            IsBusy = false;
            _streamCts?.Dispose();
            _streamCts = null;
        }
    }

    private async Task SendStreamingAsync(string userText, CancellationToken ct)
    {
        var aiMessage = AddAssistantMessage(string.Empty, isStreaming: true);
        var fullResponse = new System.Text.StringBuilder();

        await foreach (var chunk in _chatService.SendStreamingMessageAsync(_chatHistory, userText, ct))
        {
            fullResponse.Append(chunk);
            var currentText = fullResponse.ToString();

            // aiMessage implements INotifyPropertyChanged — UI updates automatically
            MainThread.BeginInvokeOnMainThread(() => aiMessage.Text = currentText);
        }

        var finalText = fullResponse.ToString();
        MainThread.BeginInvokeOnMainThread(() =>
        {
            aiMessage.Text = finalText;
            aiMessage.IsStreaming = false;
        });

        _chatHistory.Add(new ChatMessage(ChatRole.Assistant, finalText));
    }

    private async Task SendStructuredWeatherAsync(string userText, CancellationToken ct)
    {
        AddAssistantMessage("Getting weather information...", isStreaming: true);

        try
        {
            var weatherInfo = await _chatService.SendStructuredMessageAsync<WeatherInfo>(
                _chatHistory, userText, ct);

            MainThread.BeginInvokeOnMainThread(() =>
            {
                // Remove the "getting weather" placeholder
                if (Messages.LastOrDefault() is { IsAssistant: true } placeholder)
                    Messages.Remove(placeholder);

                if (weatherInfo is not null)
                {
                    var formattedWeather =
                        $"🌤 **Weather Report**\n\n" +
                        $"📍 {weatherInfo.Location}\n" +
                        $"🌡 {weatherInfo.TemperatureCelsius:F1}°C ({weatherInfo.TemperatureFahrenheit:F1}°F)\n" +
                        $"☁ {weatherInfo.Conditions}\n" +
                        $"💨 Wind: {weatherInfo.WindSpeedKmh:F1} km/h\n" +
                        $"💧 Humidity: {weatherInfo.HumidityPercent}%\n\n" +
                        $"{weatherInfo.Summary}";

                    AddAssistantMessage(formattedWeather);
                    _chatHistory.Add(new ChatMessage(ChatRole.Assistant, formattedWeather));
                }
                else
                {
                    AddAssistantMessage("Could not retrieve weather information.");
                    _chatHistory.Add(new ChatMessage(ChatRole.Assistant, "Could not retrieve weather information."));
                }
            });
        }
        catch
        {
            // Fall back to streaming if structured output fails
            MainThread.BeginInvokeOnMainThread(() =>
            {
                if (Messages.LastOrDefault() is { IsAssistant: true } placeholder)
                    Messages.Remove(placeholder);
            });
            await SendStreamingAsync(userText, ct);
        }
    }

    private void StopGeneration()
    {
        _streamCts?.Cancel();
    }

    // ── Message helpers ───────────────────────────────────────────────────────

    private void AddUserMessage(string text) =>
        MainThread.BeginInvokeOnMainThread(() =>
            Messages.Add(new ChatMessageViewModel(isUser: true, text)));

    private ChatMessageViewModel AddAssistantMessage(string text, bool isStreaming = false)
    {
        var msg = new ChatMessageViewModel(isUser: false, text) { IsStreaming = isStreaming };
        MainThread.BeginInvokeOnMainThread(() => Messages.Add(msg));
        return msg;
    }

    private void AddSystemInfo(string text) =>
        MainThread.BeginInvokeOnMainThread(() =>
            Messages.Add(new ChatMessageViewModel(isUser: false, text)));

    // ── INotifyPropertyChanged ────────────────────────────────────────────────

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnPropertyChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        OnPropertyChanged(name);
        return true;
    }
}
