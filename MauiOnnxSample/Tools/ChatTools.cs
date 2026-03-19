using System.ComponentModel;
using MauiOnnxSample.Models;
using MauiOnnxSample.Services;
using Microsoft.Extensions.AI;

namespace MauiOnnxSample.Tools;

/// <summary>
/// Provides AI tool functions that can be called by the model.
/// Each method is decorated with description attributes used by the AI to understand when to call them.
/// </summary>
public class ChatTools
{
    private readonly WeatherService _weatherService;

    public ChatTools(WeatherService weatherService)
    {
        _weatherService = weatherService;
    }

    /// <summary>Gets the device's current GPS location (latitude and longitude).</summary>
    [Description("Gets the user's current GPS location as latitude and longitude coordinates.")]
    public async Task<string> GetCurrentLocation()
    {
        try
        {
            var location = await Geolocation.GetLocationAsync(new GeolocationRequest
            {
                DesiredAccuracy = GeolocationAccuracy.Medium,
                Timeout = TimeSpan.FromSeconds(10)
            });

            if (location is null)
                return "Unable to determine location. Location services may be disabled.";

            return $"Current location: Latitude={location.Latitude:F4}, Longitude={location.Longitude:F4}";
        }
        catch (FeatureNotSupportedException)
        {
            return "GPS is not supported on this device.";
        }
        catch (FeatureNotEnabledException)
        {
            return "GPS is not enabled. Please enable location services.";
        }
        catch (PermissionException)
        {
            return "Location permission was denied. Please grant location access in app settings.";
        }
        catch (Exception ex)
        {
            return $"Failed to get location: {ex.Message}";
        }
    }

    /// <summary>Gets the current weather for a given latitude and longitude.</summary>
    [Description("Fetches current weather conditions (temperature, humidity, wind, conditions) for given GPS coordinates using the free Open-Meteo API.")]
    public async Task<string> GetWeather(
        [Description("The latitude coordinate (e.g. 47.6062 for Seattle)")]
        double latitude,
        [Description("The longitude coordinate (e.g. -122.3321 for Seattle)")]
        double longitude)
    {
        try
        {
            var (tempC, humidity, windKmh, conditions) =
                await _weatherService.GetCurrentWeatherAsync(latitude, longitude);

            double tempF = tempC * 9.0 / 5.0 + 32.0;

            return $"Weather at ({latitude:F4}, {longitude:F4}): " +
                   $"{tempC:F1}°C ({tempF:F1}°F), {conditions}. " +
                   $"Humidity: {humidity}%. Wind: {windKmh:F1} km/h.";
        }
        catch (Exception ex)
        {
            return $"Failed to fetch weather: {ex.Message}";
        }
    }

    /// <summary>Switches the app between dark, light, or system theme.</summary>
    [Description("Changes the app's visual theme. Valid values: 'dark', 'light', 'system'.")]
    public Task<string> SwitchTheme(
        [Description("The theme to switch to: 'dark', 'light', or 'system'")]
        string theme)
    {
        string result;
        var normalizedTheme = theme.Trim().ToLowerInvariant();

        AppTheme appTheme = normalizedTheme switch
        {
            "dark" => AppTheme.Dark,
            "light" => AppTheme.Light,
            _ => AppTheme.Unspecified  // system
        };

        if (normalizedTheme != "dark" && normalizedTheme != "light" && normalizedTheme != "system")
        {
            result = $"Unknown theme '{theme}'. Use 'dark', 'light', or 'system'.";
            return Task.FromResult(result);
        }

        MainThread.BeginInvokeOnMainThread(() =>
        {
            if (Application.Current is not null)
                Application.Current.UserAppTheme = appTheme;
        });

        result = normalizedTheme == "system"
            ? "Switched to system theme (follows OS setting)."
            : $"Switched to {normalizedTheme} theme.";

        return Task.FromResult(result);
    }

    /// <summary>Creates the list of AIFunction tools for use in ChatOptions.</summary>
    public IList<AITool> CreateAITools() =>
    [
        AIFunctionFactory.Create(GetCurrentLocation, "GetCurrentLocation",
            "Gets the user's current GPS location as latitude and longitude coordinates."),
        AIFunctionFactory.Create((double lat, double lon) => GetWeather(lat, lon),
            "GetWeather",
            "Fetches current weather conditions for given GPS latitude and longitude coordinates."),
        AIFunctionFactory.Create((string theme) => SwitchTheme(theme),
            "SwitchTheme",
            "Switches the app theme to 'dark', 'light', or 'system'."),
    ];
}
