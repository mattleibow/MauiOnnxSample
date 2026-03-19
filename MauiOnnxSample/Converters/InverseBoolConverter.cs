using System.Globalization;

namespace MauiOnnxSample.Converters;

/// <summary>Converts a boolean value to its inverse (true → false, false → true).</summary>
public class InverseBoolConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is bool b ? !b : false;

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is bool b ? !b : false;
}
