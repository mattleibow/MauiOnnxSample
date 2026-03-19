namespace MauiOnnxSample.Services;

/// <summary>Handles loading the ONNX model from app assets into a usable local path.</summary>
public interface IModelService
{
    /// <summary>Gets whether the model has been successfully loaded and is ready to use.</summary>
    bool IsModelReady { get; }

    /// <summary>Gets the local filesystem path to the model directory (after extraction).</summary>
    string? ModelPath { get; }

    /// <summary>
    /// Prepares the model by extracting it from app assets to local storage if necessary.
    /// Reports progress via the supplied callback.
    /// </summary>
    Task<bool> PrepareModelAsync(IProgress<string>? progress = null, CancellationToken cancellationToken = default);
}
