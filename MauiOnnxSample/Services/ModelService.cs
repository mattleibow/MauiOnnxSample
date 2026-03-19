using Microsoft.Extensions.Logging;

namespace MauiOnnxSample.Services;

/// <summary>
/// Manages the ONNX model lifecycle: detects whether model files are available, extracts them
/// from MAUI assets to app data storage on first run, and provides the ready directory path.
///
/// Priority order for finding the model:
///   1. Already-extracted model in AppDataDirectory (fastest, skips re-extraction)
///   2. Development override path: ~/Documents/phi-4-mini/ (for dev without bundled assets)
///   3. MAUI assets embedded in the app bundle (production path, copies to AppDataDirectory)
/// </summary>
public class ModelService : IModelService
{
    private const string ModelFolderName = "phi-4-mini";
    private const string AssetPrefix = "Models/" + ModelFolderName + "/";

    /// <summary>
    /// Development override path. If model files are present here, they are used directly
    /// without being copied to AppDataDirectory. This avoids embedding the ~2.8 GB model
    /// in the app bundle during development and speeds up iteration.
    ///
    /// To use: place all model files (genai_config.json, model.onnx, model.onnx.data,
    /// tokenizer.json, tokenizer_config.json) in ~/Documents/phi-4-mini/
    /// </summary>
    private static readonly string DevModelPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "phi-4-mini");

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
            // 1. Check already-extracted AppDataDirectory path
            var destDir = Path.Combine(FileSystem.AppDataDirectory, "Models", ModelFolderName);
            Directory.CreateDirectory(destDir);

            if (IsModelDirectoryReady(destDir))
            {
                _logger.LogInformation("Model already extracted at {Path}", destDir);
                SetReady(destDir, progress);
                return true;
            }

            // 2. Development override: use ~/Documents/phi-3.5-mini/ directly if present
            if (IsModelDirectoryReady(DevModelPath))
            {
                _logger.LogInformation("Using dev model at {Path}", DevModelPath);
                SetReady(DevModelPath, progress);
                return true;
            }

            // 3. Extract from MAUI assets (production path)
            progress?.Report("Extracting model from app bundle...");
            _logger.LogInformation("Extracting model assets to {Path}", destDir);

            var anyExtracted = await ExtractAllAssetsAsync(destDir, progress, cancellationToken);

            if (IsModelDirectoryReady(destDir))
            {
                SetReady(destDir, progress);
                return true;
            }

            var hint = IsModelDirectoryReady(DevModelPath)
                ? "Dev path exists but is incomplete."
                : $"Run scripts/download-model.sh and place files in {DevModelPath}";

            _logger.LogWarning("Model not ready after extraction attempt. {Hint}", hint);
            progress?.Report($"⚠️ Model not found. {hint}");
            return false;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to prepare model");
            progress?.Report($"Error: {ex.Message}");
            return false;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private void SetReady(string path, IProgress<string>? progress)
    {
        _modelPath = path;
        IsModelReady = true;
        progress?.Report("Model ready.");
        _logger.LogInformation("Model ready at {Path}", path);
    }

    /// <summary>
    /// A model directory is "ready" when it contains genai_config.json and at least one .onnx file.
    /// onnxruntime-genai resolves the exact filename from genai_config.json at load time.
    /// </summary>
    private static bool IsModelDirectoryReady(string directory) =>
        Directory.Exists(directory) &&
        File.Exists(Path.Combine(directory, "genai_config.json")) &&
        Directory.EnumerateFiles(directory, "*.onnx").Any();

    /// <summary>
    /// Enumerates known asset filenames for the model and copies any that exist in the
    /// app bundle to the destination directory.
    /// </summary>
    private async Task<bool> ExtractAllAssetsAsync(string destDir, IProgress<string>? progress, CancellationToken ct)
    {
        // These are all the files expected in the Phi-4-mini-instruct ONNX package.
        // onnxruntime-genai resolves the actual .onnx filename from genai_config.json.
        var candidates = new[]
        {
            "genai_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "config.json",
            "configuration_phi3.py",
            "merges.txt",
            "vocab.json",
            // Phi-4-mini uses standard model.onnx naming
            "model.onnx",
            "model.onnx.data",
        };

        var anyFound = false;
        foreach (var fileName in candidates)
        {
            ct.ThrowIfCancellationRequested();
            var dest = Path.Combine(destDir, fileName);
            if (File.Exists(dest))
            {
                anyFound = true;
                continue;
            }

            var assetPath = AssetPrefix + fileName;
            try
            {
                using var src = await FileSystem.OpenAppPackageFileAsync(assetPath);
                using var dst = File.Create(dest);
                progress?.Report($"Extracting {fileName}...");
                await src.CopyToAsync(dst, ct);
                _logger.LogInformation("Extracted asset: {File}", fileName);
                anyFound = true;
            }
            catch (FileNotFoundException)
            {
                _logger.LogDebug("Asset not bundled: {Asset}", assetPath);
            }
        }
        return anyFound;
    }
}
