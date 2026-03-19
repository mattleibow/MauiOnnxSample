using Microsoft.Extensions.Logging;
using MauiOnnxSample.Pages;
using MauiOnnxSample.Services;
using MauiOnnxSample.Tools;
using MauiOnnxSample.ViewModels;

#if DEBUG
using MauiDevFlow.Agent;
#endif

namespace MauiOnnxSample;

public static class MauiProgram
{
	public static MauiApp CreateMauiApp()
	{
		var builder = MauiApp.CreateBuilder();
		builder
			.UseMauiApp<App>()
			.ConfigureFonts(fonts =>
			{
				fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
				fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
			});

		// ── Core services ─────────────────────────────────────────────────────
		builder.Services.AddSingleton<IModelService, ModelService>();
		builder.Services.AddSingleton<FaqService>();
		builder.Services.AddSingleton<WeatherService>();
		builder.Services.AddSingleton<ChatTools>();
		builder.Services.AddSingleton<IChatService, ChatService>();

		// ── UI ────────────────────────────────────────────────────────────────
		builder.Services.AddTransient<ChatViewModel>();
		builder.Services.AddTransient<ChatPage>();

#if DEBUG
		builder.Logging.AddDebug();
		builder.Logging.SetMinimumLevel(LogLevel.Debug);
		builder.AddMauiDevFlowAgent();
#endif

		return builder.Build();
	}
}
