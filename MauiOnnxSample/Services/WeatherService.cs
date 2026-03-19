using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace MauiOnnxSample.Services;

/// <summary>Raw weather data from the Open-Meteo API response.</summary>
internal sealed class OpenMeteoResponse
{
    [JsonPropertyName("current")]
    public OpenMeteoCurrent? Current { get; set; }
}

internal sealed class OpenMeteoCurrent
{
    [JsonPropertyName("temperature_2m")]
    public double Temperature2m { get; set; }

    [JsonPropertyName("relative_humidity_2m")]
    public int RelativeHumidity2m { get; set; }

    [JsonPropertyName("wind_speed_10m")]
    public double WindSpeed10m { get; set; }

    [JsonPropertyName("weather_code")]
    public int WeatherCode { get; set; }
}

/// <summary>Provides weather data from the free Open-Meteo API (no API key required).</summary>
public class WeatherService : IDisposable
{
    private readonly HttpClient _httpClient;

    public WeatherService()
    {
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("MauiOnnxSample/1.0");
        _httpClient.Timeout = TimeSpan.FromSeconds(15);
    }

    /// <summary>Fetches current weather for the given coordinates.</summary>
    public async Task<(double temperatureCelsius, int humidityPercent, double windSpeedKmh, string conditions)>
        GetCurrentWeatherAsync(double latitude, double longitude, CancellationToken cancellationToken = default)
    {
        var url = $"https://api.open-meteo.com/v1/forecast" +
                  $"?latitude={latitude:F4}&longitude={longitude:F4}" +
                  $"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code" +
                  $"&wind_speed_unit=kmh";

        var response = await _httpClient.GetFromJsonAsync<OpenMeteoResponse>(url, cancellationToken);

        if (response?.Current is null)
            throw new InvalidOperationException("No weather data returned from Open-Meteo API.");

        var current = response.Current;
        var conditions = WmoCodeToConditions(current.WeatherCode);

        return (current.Temperature2m, current.RelativeHumidity2m, current.WindSpeed10m, conditions);
    }

    /// <summary>Maps WMO weather interpretation codes to human-readable conditions.</summary>
    private static string WmoCodeToConditions(int code) => code switch
    {
        0 => "Clear sky",
        1 => "Mainly clear",
        2 => "Partly cloudy",
        3 => "Overcast",
        45 or 48 => "Foggy",
        51 or 53 or 55 => "Drizzle",
        61 or 63 or 65 => "Rain",
        71 or 73 or 75 => "Snow",
        77 => "Snow grains",
        80 or 81 or 82 => "Rain showers",
        85 or 86 => "Snow showers",
        95 => "Thunderstorm",
        96 or 99 => "Thunderstorm with hail",
        _ => "Unknown conditions"
    };

    public void Dispose() => _httpClient.Dispose();
}
