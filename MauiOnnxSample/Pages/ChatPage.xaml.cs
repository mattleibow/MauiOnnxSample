using MauiOnnxSample.ViewModels;

namespace MauiOnnxSample.Pages;

public partial class ChatPage : ContentPage
{
    private readonly ChatViewModel _viewModel;

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
        // Scroll to the latest message when new ones are added
        if (_viewModel.Messages.Count > 0)
        {
            MessagesCollection.ScrollTo(
                _viewModel.Messages[^1],
                position: ScrollToPosition.End,
                animate: true);
        }
    }

    private void OnEditorCompleted(object? sender, EventArgs e)
    {
        if (_viewModel.SendCommand.CanExecute(null))
            _viewModel.SendCommand.Execute(null);
    }
}
