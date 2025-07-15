/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.superml.drift;

/**
 * DDM (Drift Detection Method) implementation for detecting concept drift
 * based on changes in error rate and variance.
 * 
 * Reference: Gama, J. et al. "Learning with Drift Detection" (2004)
 */
public class DDMDriftDetector {
    
    private final DriftConfig config;
    private double errorRate;
    private double errorVariance;
    private double minErrorRate;
    private double minErrorVariance;
    private int sampleCount;
    private int errorCount;
    private boolean warningFlag;
    
    // DDM parameters
    private static final double ALPHA_WARNING = 2.0;
    private static final double ALPHA_DRIFT = 3.0;
    
    public DDMDriftDetector(DriftConfig config) {
        this.config = config;
        reset();
    }
    
    /**
     * Process a new prediction and detect drift.
     * @param isError Whether the prediction was incorrect
     * @return DDM drift detection result
     */
    public DDMResult detectDrift(boolean isError) {
        sampleCount++;
        if (isError) {
            errorCount++;
        }
        
        // Update error rate and variance
        errorRate = (double) errorCount / sampleCount;
        errorVariance = Math.sqrt(errorRate * (1 - errorRate) / sampleCount);
        
        // Track minimum error rate and variance (best performance point)
        if (sampleCount > config.getMinSamplesForDDM() && 
            (errorRate + errorVariance) < (minErrorRate + minErrorVariance)) {
            minErrorRate = errorRate;
            minErrorVariance = errorVariance;
        }
        
        // Check for warning level
        boolean newWarning = false;
        if (sampleCount > config.getMinSamplesForDDM()) {
            double warningThreshold = minErrorRate + minErrorVariance + 
                ALPHA_WARNING * Math.sqrt(minErrorVariance);
            newWarning = (errorRate + errorVariance) > warningThreshold;
        }
        
        // Check for drift level
        boolean isDrift = false;
        if (sampleCount > config.getMinSamplesForDDM()) {
            double driftThreshold = minErrorRate + minErrorVariance + 
                ALPHA_DRIFT * Math.sqrt(minErrorVariance);
            isDrift = (errorRate + errorVariance) > driftThreshold;
        }
        
        // Update warning flag
        if (isDrift) {
            // Reset after drift detection
            reset();
        } else {
            warningFlag = newWarning;
        }
        
        return new DDMResult(isDrift, newWarning, errorRate, errorVariance, sampleCount);
    }
    
    /**
     * Reset DDM detector state.
     */
    public void reset() {
        errorRate = 0.0;
        errorVariance = 1.0;
        minErrorRate = Double.MAX_VALUE;
        minErrorVariance = Double.MAX_VALUE;
        sampleCount = 0;
        errorCount = 0;
        warningFlag = false;
    }
    
    public static class DDMResult {
        public final boolean isDrift;
        public final boolean isWarning;
        public final double errorRate;
        public final double errorVariance;
        public final int sampleCount;
        
        public DDMResult(boolean isDrift, boolean isWarning, double errorRate, 
                        double errorVariance, int sampleCount) {
            this.isDrift = isDrift;
            this.isWarning = isWarning;
            this.errorRate = errorRate;
            this.errorVariance = errorVariance;
            this.sampleCount = sampleCount;
        }
    }
}
