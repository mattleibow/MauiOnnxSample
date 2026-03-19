using Microsoft.Extensions.Logging;

namespace MauiOnnxSample.Services;

/// <summary>
/// Manages the ONNX model lifecycle: detects whether model files are available as MAUI assets,
/// extracts them to app data storage on first run, and provides the ready path.
/// </summary>
public class ModelService : IModelService
{
    private const string ModelFolderName = "phi-3.5-mini";
    private const string AssetPrefix = "Models/" + ModelFolderName + "/";

    /// <summary>Required model files that must be present for the model to function.</summary>
    private static readonly string[] RequiredModelFiles =
    [
        "model.onnx",
        "genai_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    private readonly ILogger<ModelService> _logger;
    private string? _modelPath;

    public ModelService(ILogger<ModelService> logger)
    {
        _logger = logger;
    }

    public bool IsModelReady { get; private set; }

    public string? ModelPath => _modelPath;

    public async Task<bool> PrepareModelAsync(IProgress<string>? progress = null, CancellationToken cancellationToken = default)
    {
        try
        {
            var destDir = Path.Combine(FileSystem.AppDataDirectory, "Models", ModelFolderName);
            Directory.CreateDirectory(destDir);

            // Check if all required files are already extracted
            if (AreAllModelFilesPresent(destDir))
            {
                _logger.LogInformation("Model already extracted at {Path}", destDir);
                _modelPath = destDir;
                IsModelReady = true;
                progress?.Report("Model ready.");
                return true;
            }

            progress?.Report("Extracting model from app package...");
            _logger.LogInformation("Extracting model to {Path}", destDir);

            // Attempt to list and copy assets from the app package
            foreach (var fileName in RequiredModelFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var assetPath = AssetPrefix + fileName;
                var destPath = Path.Combine(destDir, fileName);

                if (File.Exists(destPath))
                {
                    _logger.LogDebug("Skipping already-extracted {File}", fileName);
                    continue;
                }

                try
                {
                    using var assetStream = await FileSystem.OpenAppPackageFileAsync(assetPath);
                    using var fileStream = File.Create(destPath);
                    progress?.Report($"Extracting {fileName}...");
                    await assetStream.CopyToAsync(fileStream, cancellationToken);
                    _logger.LogInformation("Extracted {File}", fileName);
                }
                catch (FileNotFoundException)
                {
                    _logger.LogWarning("Asset not found: {Asset} — model may not be bundled", assetPath);
                }
            }

            // Also copy any additional files (like model.onnx.data which is large)
            await TryCopyLargeAssetAsync("model.onnx.data", destDir, progress, cancellationToken);
            await TryCopyLargeAssetAsync("special_tokens_map.json", destDir, progress, cancellationToken);

            if (AreAllModelFilesPresent(destDir))
            {
                _modelPath = destDir;
                IsModelReady = true;
                progress?.Report("Model extraction complete.");
                _logger.LogInformation("Model ready at {Path}", destDir);
                return true;
            }

            _logger.LogWarning("Model files missing. Run the download script to obtain model files.");
            progress?.Report("Model files not found. See README.txt in Resources/Raw/Models/phi-3.5-mini/");
            return false;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to prepare model");
            progress?.Report($"Error preparing model: {ex.Message}");
            return false;
        }
    }

    private async Task TryCopyLargeAssetAsync(string fileName, string destDir, IProgress<string>? progress, CancellationToken cancellationToken)
    {
        var destPath = Path.Combine(destDir, fileName);
        if (File.Exists(destPath))
            return;

        var assetPath = AssetPrefix + fileName;
        try
        {
            using var assetStream = await FileSystem.OpenAppPackageFileAsync(assetPath);
            using var fileStream = File.Create(destPath);
            progress?.Report($"Extracting {fileName} (large file, please wait)...");
            await assetStream.CopyToAsync(fileStream, cancellationToken);
            _logger.LogInformation("Extracted large file {File}", fileName);
        }
        catch (FileNotFoundException)
        {
            _logger.LogDebug("Optional asset not found: {Asset}", assetPath);
        }
    }

    private static bool AreAllModelFilesPresent(string directory) =>
        RequiredModelFiles.All(f => File.Exists(Path.Combine(directory, f)));
}
