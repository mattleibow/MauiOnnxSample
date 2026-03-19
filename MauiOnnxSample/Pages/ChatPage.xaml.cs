using MauiOnnxSample.Models;
using MauiOnnxSample.ViewModels;

namespace MauiOnnxSample.Pages;

/// <summary>
/// Chat page. Uses ScrollView+VerticalStackLayout instead of CollectionView to avoid
/// a SIGSEGV crash in libswiftObservation.dylib triggered by CollectionView's compiled
/// bindings (x:DataType) on MAUI 11 preview / Mac Catalyst.
///
/// Message views are added programmatically via AddMessageView(). The ViewModel still
/// owns the ChatMessageViewModel objects; the page simply renders them.
/// </summary>
public partial class ChatPage : ContentPage
{
    private readonly ChatViewModel _viewModel;

    // Tracks the live Label inside a streaming AI bubble so we can update its text
    private readonly Dictionary<ChatMessageViewModel, Label> _streamingLabels = new();

    public ChatPage(ChatViewModel viewModel)
    {
        InitializeComponent();
        _viewModel = viewModel;
        BindingContext = viewModel;

        _viewModel.Messages.CollectionChanged += OnMessagesChanged;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        if (_viewModel.IsModelLoading)
            await _viewModel.PrepareModelAsync();
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        _viewModel.Messages.CollectionChanged -= OnMessagesChanged;
    }

    private void OnMessagesChanged(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
    {
        if (e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Add && e.NewItems is not null)
        {
            foreach (ChatMessageViewModel msg in e.NewItems)
                AddMessageView(msg);
        }
        else if (e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Remove && e.OldItems is not null)
        {
            // Remove by tag (we set each view's tag to the ChatMessageViewModel)
            foreach (ChatMessageViewModel msg in e.OldItems)
            {
                var toRemove = MessagesContainer.Children
                    .OfType<View>()
                    .FirstOrDefault(v => v.BindingContext == msg);
                if (toRemove is not null)
                    MessagesContainer.Children.Remove(toRemove);
                _streamingLabels.Remove(msg);
            }
        }
        else if (e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Reset)
        {
            MessagesContainer.Children.Clear();
            _streamingLabels.Clear();
        }

        ScrollToBottom();
    }

    private void AddMessageView(ChatMessageViewModel msg)
    {
        View view = msg.IsUser ? BuildUserBubble(msg) : BuildAiBubble(msg);
        view.BindingContext = msg; // used for removal lookup
        MessagesContainer.Children.Add(view);
    }

    private View BuildUserBubble(ChatMessageViewModel msg)
    {
        var label = new Label
        {
            Text = msg.Text,
            TextColor = Colors.White,
            FontSize = 15,
            LineBreakMode = LineBreakMode.WordWrap,
        };

        var border = new Border
        {
            BackgroundColor = Application.Current?.RequestedTheme == AppTheme.Dark
                ? Color.FromArgb("#7B5EA7") : Color.FromArgb("#512BD4"),
            StrokeThickness = 0,
            StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = new CornerRadius(18, 18, 4, 18) },
            Padding = new Thickness(14, 10),
            Margin = new Thickness(60, 4, 12, 4),
            HorizontalOptions = LayoutOptions.End,
            MaximumWidthRequest = 320,
            Content = label,
        };

        // Wire up live text updates (shouldn't happen for user messages, but future-proof)
        msg.PropertyChanged += (_, e) => { if (e.PropertyName == nameof(msg.Text)) label.Text = msg.Text; };

        return new VerticalStackLayout
        {
            HorizontalOptions = LayoutOptions.End,
            Children =
            {
                new Label
                {
                    Text = msg.AuthorLabel,
                    FontSize = 11,
                    TextColor = Application.Current?.RequestedTheme == AppTheme.Dark
                        ? Color.FromArgb("#9B9B9B") : Color.FromArgb("#6B6B6B"),
                    Margin = new Thickness(16, 0, 16, 2),
                    HorizontalOptions = LayoutOptions.End,
                },
                border,
            }
        };
    }

    private View BuildAiBubble(ChatMessageViewModel msg)
    {
        var textLabel = new Label
        {
            Text = msg.Text,
            FontSize = 15,
            LineBreakMode = LineBreakMode.WordWrap,
        };
        textLabel.SetAppThemeColor(Label.TextColorProperty,
            Color.FromArgb("#111111"), Colors.White);

        // Save label reference so streaming updates can reach it
        _streamingLabels[msg] = textLabel;

        var spinner = new ActivityIndicator
        {
            IsRunning = msg.IsStreaming,
            IsVisible = msg.IsStreaming,
            Color = Color.FromArgb("#512BD4"),
            WidthRequest = 14,
            HeightRequest = 14,
            Margin = new Thickness(6, 0, 0, 0),
            VerticalOptions = LayoutOptions.Center,
        };

        var border = new Border
        {
            StrokeThickness = 1,
            StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = new CornerRadius(18, 18, 18, 4) },
            Padding = new Thickness(14, 10),
            Margin = new Thickness(12, 4, 60, 4),
            HorizontalOptions = LayoutOptions.Start,
            MaximumWidthRequest = 320,
            Content = textLabel,
        };
        border.SetAppThemeColor(Border.BackgroundColorProperty, Colors.White, Color.FromArgb("#2D2D2D"));
        border.SetAppThemeColor(Border.StrokeProperty, Color.FromArgb("#E0E0E0"), Color.FromArgb("#555555"));

        // Wire up live property updates for streaming
        msg.PropertyChanged += (_, e) =>
        {
            MainThread.BeginInvokeOnMainThread(() =>
            {
                if (e.PropertyName == nameof(msg.Text))
                {
                    textLabel.Text = msg.Text;
                }
                else if (e.PropertyName == nameof(msg.IsStreaming))
                {
                    spinner.IsRunning = msg.IsStreaming;
                    spinner.IsVisible = msg.IsStreaming;
                }
            });
        };

        return new VerticalStackLayout
        {
            HorizontalOptions = LayoutOptions.Start,
            Children =
            {
                new Grid
                {
                    ColumnDefinitions = [new ColumnDefinition(GridLength.Auto), new ColumnDefinition(GridLength.Auto)],
                    Margin = new Thickness(16, 0, 0, 0),
                    Children =
                    {
                        new Label
                        {
                            Text = msg.AuthorLabel,
                            FontSize = 11,
                            TextColor = Application.Current?.RequestedTheme == AppTheme.Dark
                                ? Color.FromArgb("#9B9B9B") : Color.FromArgb("#6B6B6B"),
                        },
                        spinner.WithGridColumn(1),
                    }
                },
                border,
            }
        };
    }

    private void ScrollToBottom()
    {
        if (MessagesContainer.Children.Count > 0)
            MessagesScrollView.ScrollToAsync(0, double.MaxValue, animated: true);
    }

    private void OnEditorCompleted(object? sender, EventArgs e)
    {
        if (_viewModel.SendCommand.CanExecute(null))
            _viewModel.SendCommand.Execute(null);
    }
}

// Small extension to set Grid.Column without XAML
file static class ViewExtensions
{
    public static T WithGridColumn<T>(this T view, int column) where T : View
    {
        Grid.SetColumn(view, column);
        return view;
    }
}
