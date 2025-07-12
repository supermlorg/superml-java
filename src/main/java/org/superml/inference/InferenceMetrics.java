package org.superml.inference;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicInteger;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Performance metrics for model inference operations.
 * Tracks inference statistics, timing, and error rates.
 */
public class InferenceMetrics {
    private final String modelId;
    private final AtomicLong totalInferences = new AtomicLong(0);
    private final AtomicLong totalSamples = new AtomicLong(0);
    private final AtomicLong totalInferenceTimeNanos = new AtomicLong(0);
    private final AtomicInteger errorCount = new AtomicInteger(0);
    private final LocalDateTime createdAt;
    
    // Statistics tracking
    private volatile long minInferenceTimeNanos = Long.MAX_VALUE;
    private volatile long maxInferenceTimeNanos = Long.MIN_VALUE;
    private volatile long lastInferenceTimeNanos = 0;
    
    /**
     * Create metrics for a model.
     * @param modelId model identifier
     */
    public InferenceMetrics(String modelId) {
        this.modelId = modelId;
        this.createdAt = LocalDateTime.now();
    }
    
    /**
     * Record a successful inference operation.
     * @param sampleCount number of samples processed
     * @param inferenceTimeNanos time taken in nanoseconds
     */
    public void recordInference(int sampleCount, long inferenceTimeNanos) {
        totalInferences.incrementAndGet();
        totalSamples.addAndGet(sampleCount);
        totalInferenceTimeNanos.addAndGet(inferenceTimeNanos);
        lastInferenceTimeNanos = inferenceTimeNanos;
        
        // Update min/max (thread-safe updates)
        updateMinTime(inferenceTimeNanos);
        updateMaxTime(inferenceTimeNanos);
    }
    
    /**
     * Record an error during inference.
     */
    public void recordError() {
        errorCount.incrementAndGet();
    }
    
    /**
     * Get total number of inference operations.
     * @return total inferences
     */
    public long getTotalInferences() {
        return totalInferences.get();
    }
    
    /**
     * Get total number of samples processed.
     * @return total samples
     */
    public long getTotalSamples() {
        return totalSamples.get();
    }
    
    /**
     * Get total inference time in milliseconds.
     * @return total time in ms
     */
    public double getTotalInferenceTimeMs() {
        return totalInferenceTimeNanos.get() / 1_000_000.0;
    }
    
    /**
     * Get average inference time per operation in milliseconds.
     * @return average time in ms
     */
    public double getAverageInferenceTimeMs() {
        long inferences = totalInferences.get();
        if (inferences == 0) return 0.0;
        return (totalInferenceTimeNanos.get() / 1_000_000.0) / inferences;
    }
    
    /**
     * Get average time per sample in microseconds.
     * @return average time per sample in μs
     */
    public double getAverageTimePerSampleMicros() {
        long samples = totalSamples.get();
        if (samples == 0) return 0.0;
        return (totalInferenceTimeNanos.get() / 1_000.0) / samples;
    }
    
    /**
     * Get minimum inference time in milliseconds.
     * @return min time in ms
     */
    public double getMinInferenceTimeMs() {
        return minInferenceTimeNanos == Long.MAX_VALUE ? 0.0 : minInferenceTimeNanos / 1_000_000.0;
    }
    
    /**
     * Get maximum inference time in milliseconds.
     * @return max time in ms
     */
    public double getMaxInferenceTimeMs() {
        return maxInferenceTimeNanos == Long.MIN_VALUE ? 0.0 : maxInferenceTimeNanos / 1_000_000.0;
    }
    
    /**
     * Get last inference time in milliseconds.
     * @return last time in ms
     */
    public double getLastInferenceTimeMs() {
        return lastInferenceTimeNanos / 1_000_000.0;
    }
    
    /**
     * Get error count.
     * @return number of errors
     */
    public int getErrorCount() {
        return errorCount.get();
    }
    
    /**
     * Get error rate as percentage.
     * @return error rate (0-100)
     */
    public double getErrorRate() {
        long inferences = totalInferences.get();
        if (inferences == 0) return 0.0;
        return (errorCount.get() * 100.0) / (inferences + errorCount.get());
    }
    
    /**
     * Get throughput in samples per second.
     * @return samples per second
     */
    public double getThroughputSamplesPerSecond() {
        double totalTimeSeconds = totalInferenceTimeNanos.get() / 1_000_000_000.0;
        if (totalTimeSeconds == 0.0) return 0.0;
        return totalSamples.get() / totalTimeSeconds;
    }
    
    /**
     * Get throughput in inferences per second.
     * @return inferences per second
     */
    public double getThroughputInferencesPerSecond() {
        double totalTimeSeconds = totalInferenceTimeNanos.get() / 1_000_000_000.0;
        if (totalTimeSeconds == 0.0) return 0.0;
        return totalInferences.get() / totalTimeSeconds;
    }
    
    /**
     * Get model ID.
     * @return model identifier
     */
    public String getModelId() {
        return modelId;
    }
    
    /**
     * Get creation timestamp.
     * @return creation time
     */
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    /**
     * Reset all metrics.
     */
    public void reset() {
        totalInferences.set(0);
        totalSamples.set(0);
        totalInferenceTimeNanos.set(0);
        errorCount.set(0);
        minInferenceTimeNanos = Long.MAX_VALUE;
        maxInferenceTimeNanos = Long.MIN_VALUE;
        lastInferenceTimeNanos = 0;
    }
    
    /**
     * Get formatted summary of metrics.
     * @return metrics summary
     */
    public String getSummary() {
        return String.format(
            "InferenceMetrics[%s]: " +
            "inferences=%d, samples=%d, avgTime=%.2fms, " +
            "throughput=%.1f samples/s, errors=%d (%.1f%%)",
            modelId, getTotalInferences(), getTotalSamples(), 
            getAverageInferenceTimeMs(), getThroughputSamplesPerSecond(),
            getErrorCount(), getErrorRate()
        );
    }
    
    @Override
    public String toString() {
        return getSummary();
    }
    
    // Thread-safe min/max updates
    private void updateMinTime(long time) {
        long current = minInferenceTimeNanos;
        while (time < current && !compareAndSetMinTime(current, time)) {
            current = minInferenceTimeNanos;
        }
    }
    
    private void updateMaxTime(long time) {
        long current = maxInferenceTimeNanos;
        while (time > current && !compareAndSetMaxTime(current, time)) {
            current = maxInferenceTimeNanos;
        }
    }
    
    // Simplified atomic operations (would use AtomicLong in production)
    private synchronized boolean compareAndSetMinTime(long expect, long update) {
        if (minInferenceTimeNanos == expect) {
            minInferenceTimeNanos = update;
            return true;
        }
        return false;
    }
    
    private synchronized boolean compareAndSetMaxTime(long expect, long update) {
        if (maxInferenceTimeNanos == expect) {
            maxInferenceTimeNanos = update;
            return true;
        }
        return false;
    }
}
