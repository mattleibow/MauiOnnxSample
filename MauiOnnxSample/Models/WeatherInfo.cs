using System.Text.Json.Serialization;

namespace MauiOnnxSample.Models;

/// <summary>Structured weather information returned via GetResponseAsync&lt;WeatherInfo&gt;().</summary>
public class WeatherInfo
{
    [JsonPropertyName("location")]
    public string Location { get; set; } = string.Empty;

    [JsonPropertyName("temperature_celsius")]
    public double TemperatureCelsius { get; set; }

    [JsonPropertyName("temperature_fahrenheit")]
    public double TemperatureFahrenheit { get; set; }

    [JsonPropertyName("conditions")]
    public string Conditions { get; set; } = string.Empty;

    [JsonPropertyName("wind_speed_kmh")]
    public double WindSpeedKmh { get; set; }

    [JsonPropertyName("humidity_percent")]
    public int HumidityPercent { get; set; }

    [JsonPropertyName("summary")]
    public string Summary { get; set; } = string.Empty;

    public override string ToString() =>
        $"{Location}: {TemperatureCelsius:F1}°C ({TemperatureFahrenheit:F1}°F), {Conditions}. Wind: {WindSpeedKmh} km/h. Humidity: {HumidityPercent}%. {Summary}";
}
